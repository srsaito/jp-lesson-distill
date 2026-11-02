"""Stage orchestration. Each stage caches its output under work/<date>/ and is
skipped when the output file already exists — delete a stage file to redo it."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import prompts
from .archive import archive_lesson
from .board import find_video, run_board
from .board import notes as board_notes
from .audio import clip_audio, duration_seconds, prep_audio, split_windows, window_starts
from .gemini import (
    CALL_FAILED,
    DEFAULT_MODEL,
    PASS_A_THINKING_LEVEL,
    STREAM_IDLE_TIMEOUT_S,
    RepetitionLoop,
    audio_part,
    generate,
    make_client,
    upload_audio,
)
from .models import (
    BoardNote,
    Candidate,
    CandidateList,
    Moment,
    MomentsFile,
    Relisten,
    Segment,
    TrackTranscript,
    Transcript,
    fmt_ts,
    parse_ts,
)
from .quality import (
    TrackReport,
    WindowReport,
    evaluate,
    evaluate_track,
    is_ghost,
    marker_by_window,
    marker_verdict,
)
from .tracks import identify as identify_tracks
from .tracks import FRAME_S, SOUND_DB, load_energy, relabel, snap_start
from .tracks import tracks_dir as tracks_dir_of
from .tracks import select as select_tracks

CLIP_PAD = 15.0  # seconds of context on each side of a candidate
BOARD_BEFORE, BOARD_AFTER = 30.0, 90.0  # seconds: typing trails the speech it answers (ADR-0009)
# Seconds of silence before a re-listen call is abandoned and re-rolled. Measured 2026-10-08
# on 10/5's clips: 15 healthy calls took 7–26 s end to end, and a stalled one sends nothing at
# all — so 90 s is three times the slowest healthy call, against the 300 s Pass A needs.
PASS_B_IDLE_TIMEOUT_S = 90.0
PASS_B_GIVE_UP = 2  # consecutive failed clips before Pass B stops trying the rest (jld-8rb)


@dataclass
class Config:
    recording: Path
    date: str  # YYYYMMDD
    model: str = DEFAULT_MODEL
    work_dir: Path = Path("work")
    out_dir: Path = Path("/Users/stevensaito/Docs/Vault-GeneralNotes/_raw")
    skip_pass_b: bool = False
    max_moments: int = 40
    window_minutes: float = 20.0  # 0 = transcribe the whole recording in one call
    overlap_seconds: float = 30.0
    window_attempts: int = 2  # re-rolls of a window the sanity gates reject
    stream_timeout: float = STREAM_IDLE_TIMEOUT_S  # seconds of silence before a call is abandoned
    # Pass A only; None leaves the model's own default in place. Detect and Pass B are never
    # given a level — those two stages are where thinking earns its tokens.
    pass_a_thinking: str | None = PASS_A_THINKING_LEVEL
    # Copy this lesson's non-audio outputs to OneDrive when the run ends (docs/archive.md).
    archive: bool = True
    # Read the shared screen into a board timeline (ADR-0008). `video` None = look beside the
    # recording; a lesson with no video simply has no board.
    board: bool = True
    video: Path | None = None
    # Pass A path (ADR-0010): "auto" uses the Zoom per-participant tracks when they are usable
    # and falls back to the mixed recording otherwise; "per-track" / "mixed" force one.
    tracks_mode: str = "auto"

    @property
    def lesson_date(self) -> str:
        return f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"


def _load(path: Path, schema):
    return schema.model_validate_json(path.read_text())


def _save(path: Path, obj) -> None:
    path.write_text(obj.model_dump_json(indent=2))


def run(cfg: Config) -> Path | None:
    """Run every stage, then archive whatever the run produced — even if a stage failed.

    The archive step sits in `finally` on purpose. A run that dies in window 3 has already
    paid for windows 1 and 2, and those transcripts are exactly what a re-run cannot give
    back; they are worth keeping most when something has gone wrong.
    """
    try:
        return _run(cfg)
    finally:
        if cfg.archive:
            archive_lesson(cfg.date, cfg.work_dir, cfg.recording)


def _board(cfg: Config) -> list[BoardNote]:
    """The board timeline, and the lines typed on it, for the moments to carry.

    Warn-only, like the archive: the moments are already emitted by the time this runs, and
    a screen that could not be read (no video, not macOS, a Gemini hiccup on a slide) costs
    them their board lines, nothing more. `distill board` retries it alone.

    `--no-board` does not read a screen (that is the slow part), but a board already on disk
    is still used: it is free.
    """
    cached = cfg.work_dir / cfg.date / "board" / "board.json"
    try:
        if not cfg.board:
            board = json.loads(cached.read_text()) if cached.exists() else None
        else:
            video = cfg.video or find_video(cfg.recording)
            if video is None:
                print(f"[board] no video beside {cfg.recording.name} — no board timeline "
                      "(pass --video, or run `distill board <video> --date …`)")
                return []
            board = run_board(video, cfg.date, cfg.work_dir, cfg.out_dir)
        return [BoardNote(**n) for n in board_notes(board)] if board else []
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(f"[board] WARNING: board timeline failed ({type(exc).__name__}: {exc}); carrying on "
              f"without it. Retry with `distill board <video> --date {cfg.date}`")
        return []


def _run(cfg: Config) -> Path | None:
    work = cfg.work_dir / cfg.date
    work.mkdir(parents=True, exist_ok=True)
    client = None  # created only if a Gemini stage actually runs

    # prep
    audio = work / "audio.m4a"
    if audio.exists():
        print(f"[prep] cached: {audio}")
    else:
        print(f"[prep] extracting mono audio from {cfg.recording}")
        prep_audio(cfg.recording, audio)
    total = duration_seconds(audio)
    print(f"[prep] duration {fmt_ts(total)}")

    # Which Pass A path (ADR-0010). The mixed path is the required fallback, so every way the
    # tracks can be unusable ends here with a reason, never with a failed run.
    selection = select_tracks(cfg.recording, total, cfg.tracks_mode, work)
    print(f"[tracks] {selection.path}: {selection.reason}")
    for warning in selection.warnings:
        print(f"[tracks] warning: {warning}")

    # pass A
    def get_client():
        nonlocal client
        client = client or make_client(cfg.stream_timeout)
        return client

    # A cached transcript keeps the path that made it: one made before the tracks were usable
    # (or before this code) is a mixed transcript, and still gets the label check below.
    used = _pass_a_path(work) if (work / "transcript.json").exists() else selection.path
    if used == "per-track":
        transcript = _pass_a_tracks(cfg, work, audio, total, get_client)
    else:
        transcript = _pass_a(cfg, work, audio, total, get_client)
        bounds = window_bounds(total, cfg.window_minutes * 60, cfg.overlap_seconds)
        print(f"[pass_a] {marker_by_window(transcript.segments, bounds)}")
        transcript = _relabel(work, transcript, total, bounds)

    # detect
    candidates_path = work / "candidates.json"
    if candidates_path.exists():
        print(f"[detect] cached: {candidates_path}")
    else:
        client = get_client()
        # Clips and re-listens are filed by candidate number; a new detect renumbers them.
        for stale in ("clips", "relisten"):
            shutil.rmtree(work / stale, ignore_errors=True)
        print("[detect] flagging candidate learning moments")
        prompt = prompts.DETECT.format(transcript=transcript.model_dump_json())
        cands: CandidateList = generate(client, cfg.model, [prompt], CandidateList, progress=True)
        _save(candidates_path, cands)
    cands = _load(candidates_path, CandidateList)
    ordered = sorted(cands.candidates, key=lambda c: c.priority, reverse=True)
    if len(ordered) > cfg.max_moments:
        print(f"[detect] {len(ordered)} candidates; keeping top {cfg.max_moments} by priority "
              f"(dropping {len(ordered) - cfg.max_moments} — raise --max-moments to include them)")
        ordered = ordered[: cfg.max_moments]
    else:
        print(f"[detect] {len(ordered)} candidates")

    if cfg.skip_pass_b:
        print("[pass_b] skipped (--skip-pass-b); no moments.json emitted")
        _emit_transcript(cfg, transcript)
        return None

    # pass B
    moments_path = work / "moments.json"
    if moments_path.exists():
        print(f"[pass_b] cached: {moments_path}")
    else:
        # Its own client: a re-listen answers in seconds, so it should not wait as long as a
        # twenty-minute Pass A window does before a silent call is given up on.
        client = make_client(min(cfg.stream_timeout, PASS_B_IDLE_TIMEOUT_S))
        clips_dir = work / "clips"
        clips_dir.mkdir(exist_ok=True)
        # One file per clip (jld-8rb). moments.json is only written when every clip is in, so
        # without these a call that failed on clip 16 threw away the fifteen already paid for.
        relisten_dir = work / "relisten"
        relisten_dir.mkdir(exist_ok=True)
        moments: list[Moment] = []
        failed: list[str] = []
        streak = 0  # consecutive clips whose call failed
        for i, cand in enumerate(ordered, start=1):
            t0, t1 = parse_ts(cand.t_start), parse_ts(cand.t_end)
            clip = clips_dir / f"m{i:02d}.m4a"
            if not clip.exists():
                clip_audio(audio, clip, t0 - CLIP_PAD, min(total, t1 + CLIP_PAD))
            saved = relisten_dir / f"m{i:02d}.json"
            if saved.exists():
                print(f"[pass_b] {i}/{len(ordered)} cached: {saved.name}")
                r: Relisten = _load(saved, Relisten)
            else:
                print(f"[pass_b] {i}/{len(ordered)} re-listening {cand.t_start}-{cand.t_end} ({cand.type})")
                if streak >= PASS_B_GIVE_UP:
                    failed.append(f"{i} ({cand.t_start}, not tried)")
                    continue
                try:
                    try:
                        r = _relisten(client, cfg.model, clip, cand)
                    except CALL_FAILED as exc:
                        # The stall is tied to thinking: the clip that went silent eight times
                        # running at the model's default answered in 3 s at LOW (2026-10-08).
                        # A re-listen with less thinking beats a lesson with no moments.
                        print(f"[pass_b]   no answer after retries ({type(exc).__name__}); "
                              "one more try with thinking turned down", flush=True)
                        r = _relisten(client, cfg.model, clip, cand, thinking_level="LOW")
                except CALL_FAILED as exc:
                    # Its retries are spent. One bad clip is no reason to drop the others, which
                    # are independent of it. Two in a row is the network or the service, and each
                    # further clip would wait out four idle timeouts before failing the same way.
                    streak += 1
                    failed.append(f"{i} ({cand.t_start})")
                    print(f"[pass_b]   FAILED after retries ({type(exc).__name__}); "
                          + ("stopping here, the next ones would fail the same way"
                             if streak >= PASS_B_GIVE_UP else "carrying on with the rest"), flush=True)
                    continue
                streak = 0
                _save(saved, r)
            if not r.keep:
                print(f"[pass_b]   rejected: {r.explanation}")
                continue
            moments.append(Moment(
                id=f"m{i:02d}", t_start=t0, t_end=t1, type=r.type,
                student_verbatim=r.student_verbatim, teacher_correction=r.teacher_correction,
                explanation=r.explanation, confidence=r.confidence,
            ))
        if failed:
            # No moments.json: a file missing some moments would look finished, be emitted, and
            # be cached as the answer. Stopping here costs one re-run that only redoes the gaps.
            raise SystemExit(
                f"[pass_b] {len(failed)} of {len(ordered)} clips could not be re-listened: "
                f"{', '.join(failed)}. The other {len(ordered) - len(failed)} are saved in "
                f"{relisten_dir}; run the same command again and only these are retried. "
                "No moments.json was written.")
        out = MomentsFile(
            lesson_date=cfg.lesson_date,
            source_recording=str(cfg.recording),
            model=cfg.model,
            generated_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            moments=moments,
        )
        _save(moments_path, out)
    out = _load(moments_path, MomentsFile)
    print(f"[pass_b] {len(out.moments)} moments confirmed")

    # emit
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.out_dir / f"{cfg.date}-moments.json"
    dest.write_text(out.model_dump_json(indent=2))
    _emit_transcript(cfg, transcript)
    print(f"[emit] {dest}")

    # board — after the moments are out, because reading the screen takes 10–20 minutes. Then
    # the moments are written once more, each with what went onto the screen around it (ADR-0009).
    notes = _board(cfg)
    if notes:
        for m in out.moments:
            m.board = _near(notes, m.t_start, m.t_end)
        dest.write_text(out.model_dump_json(indent=2))
        print(f"[emit] {dest} — {sum(bool(m.board) for m in out.moments)} of {len(out.moments)} "
              "moments now carry board lines")
    return dest


def window_bounds(total: float, win_s: float, overlap_s: float) -> list[tuple[float, float]]:
    """The stretch of the merged transcript each Pass A window owns (the merge's midpoint rule)."""
    starts = window_starts(total, win_s, overlap_s)
    if not starts:
        return [(0.0, total)]
    ends = [min(total, s + win_s) for s in starts]
    cuts = [(starts[i + 1] + ends[i]) / 2 for i in range(len(starts) - 1)]
    return list(zip([0.0, *cuts], [*cuts, total]))


def _relabel(work: Path, transcript: Transcript, total: float,
             bounds: list[tuple[float, float]]) -> Transcript:
    """Check the mixed transcript's speaker labels against the tracks, and fix the clear errors.

    jld-ec6: on 10/14 a whole window came back with teacher and student swapped, the address
    forms said "consistent", and the lesson lost 10 of 22 moments. Each mic track hears only its
    own speaker, so where one track clearly dominates a segment the speaker is known. Flipped
    labels are written into transcript.json (the original is kept as
    transcript.before_relabel.json) before detect reads it. Runs once per lesson; the report in
    tracks/relabel.json marks it done.
    """
    report_path = work / "tracks" / "relabel.json"
    if report_path.exists():
        print(f"[relabel] cached: {report_path}")
        return transcript
    energy = load_energy(work)
    if energy is None:
        print("[relabel] no usable per-participant tracks: speaker labels are checked by "
              "address forms only")
        return transcript
    segments, checks = relabel(transcript.segments, energy, total, bounds)
    for c in checks:
        rate = c.agreement
        verdict = ("" if rate is None or rate >= 0.5
                   else " — this window came back mostly INVERTED")
        print(f"[relabel] w{c.window:02d} {fmt_ts(c.start_s)}-{fmt_ts(c.end_s)}: "
              f"{c.agree}/{c.agree + c.flipped} labels agree with the tracks, {c.flipped} flipped, "
              f"{c.undecided} undecided (overlap or two voices in one segment), "
              f"{c.played} over playback{verdict}")
    flipped = sum(c.flipped for c in checks)
    agree = sum(c.agree for c in checks)
    print(f"[relabel] {flipped} of {len(segments)} segments relabelled from the tracks "
          f"({agree}/{agree + flipped} already agreed)")
    if flipped:
        shutil.copyfile(work / "transcript.json", work / "transcript.before_relabel.json")
        transcript = Transcript(segments=segments)
        _save(work / "transcript.json", transcript)
        if (work / "candidates.json").exists():
            print("[relabel] WARNING: candidates.json was detected from the old labels; delete it "
                  "(and moments.json) to re-detect from the relabelled transcript")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"flipped": flipped, "windows": [asdict(c) for c in checks]},
                                      indent=2))
    return transcript


def _pass_a_path(work: Path) -> str:
    """Which path made the cached transcript.json ("mixed" for one made before this was recorded)."""
    record = work / "pass_a.json"
    return json.loads(record.read_text())["path"] if record.exists() else "mixed"


def _record_pass_a_path(work: Path, path: str, fallback_windows: list[int] | None = None) -> None:
    (work / "pass_a.json").write_text(json.dumps(
        {"path": path, "fallback_windows": fallback_windows or []}, indent=2))


TRACK_PROMPTS = {"teacher": prompts.TRACK_TEACHER, "student": prompts.TRACK_STUDENT}
TRACK_SILENT_S = 1.0  # under this much sound in a window, the track is not worth a call


def _pass_a_tracks(cfg: Config, work: Path, audio: Path, total: float, get_client) -> Transcript:
    """Pass A per Zoom participant track (ADR-0010).

    Each window of Soso先生's mic and of Steven's mic is transcribed on its own, the two calls in
    parallel, so the speaker is whichever file a line came from: no labels to get wrong, and no
    two voices run together in one segment. Steven's track gets no teacher context (the measured
    choice). Windows are cut at the mix's offsets and merged by the mix's rule, and every start
    is then snapped to its own track's sound onset.

    A window whose per-track calls fail after their retries is transcribed from the mix instead,
    and its labels are repaired from the tracks: the mixed path is the required fallback, per
    window as well as per lesson. pass_a.json records which windows that happened to.
    """
    transcript_path = work / "transcript.json"
    if transcript_path.exists():
        print(f"[pass_a] cached: {transcript_path} (per-participant tracks)")
        return _load(transcript_path, Transcript)
    energy = load_energy(work)
    tracks_dir = work / "tracks"
    win_s = cfg.window_minutes * 60
    folder = tracks_dir_of(cfg.recording)
    found = identify_tracks(folder)

    role_windows: dict[str, list[tuple[Path, float]]] = {}
    for role in TRACK_PROMPTS:
        prepped = tracks_dir / f"{role}.m4a"
        if not prepped.exists():
            print(f"[pass_a] preparing the {role} track ({found[role].name})")
            prep_audio(found[role], prepped)
        role_windows[role] = split_windows(prepped, tracks_dir / f"{role}_windows", win_s=win_s,
                                           overlap_s=cfg.overlap_seconds, total=total)
    mix_windows = split_windows(audio, work / "windows", win_s=win_s,
                                overlap_s=cfg.overlap_seconds, total=total)
    n = len(mix_windows)
    print(f"[pass_a] per-participant tracks: {n} window(s), teacher and student transcribed "
          "separately and in parallel")

    parts: list[list[Segment]] = []
    offsets: list[float] = []
    fallback: list[int] = []
    for i, (mix_path, offset) in enumerate(mix_windows, start=1):
        tag = f"window {i}/{n}"
        span = f"{fmt_ts(offset)}-{fmt_ts(min(total, offset + win_s) if win_s else total)}"
        window_s = duration_seconds(mix_path) if n > 1 else total
        wpath = tracks_dir / f"transcript_w{i:02d}.json"
        if wpath.exists():
            print(f"[pass_a] {tag} cached: {wpath}")
        else:
            try:
                segments = _transcribe_tracks_window(cfg, get_client, role_windows, i, tag, span,
                                                     tracks_dir, energy, offset, window_s)
                _save(wpath, Transcript(segments=segments))
            except Exception as exc:  # noqa: BLE001 — any failure here has a fallback
                print(f"[pass_a] {tag}: per-track transcription FAILED ({type(exc).__name__}: "
                      f"{exc}); transcribing this window from the mixed recording instead")
                wt = _transcribe_window(cfg, get_client, mix_path, tag, span, window_s,
                                        work / f"transcript_w{i:02d}.json")
                _save(wpath, wt)
                (tracks_dir / f"transcript_w{i:02d}.fallback").write_text("mixed\n")
        if (tracks_dir / f"transcript_w{i:02d}.fallback").exists():
            fallback.append(i)
        parts.append(_load(wpath, Transcript).segments)
        offsets.append(offset)

    merged = merge_windows(parts, offsets, win_s=win_s, total=total, overlap_s=cfg.overlap_seconds)
    segments = merged.segments
    if fallback and energy is not None:
        # Windows that fell back are mixed transcripts: repair their labels like the mixed path.
        bounds = window_bounds(total, win_s, cfg.overlap_seconds)
        fixed, _ = relabel(segments, energy, total, bounds)
        own = [bounds[w - 1] for w in fallback]
        segments = [f if any(a <= parse_ts(s.start) < b for a, b in own) else s
                    for s, f in zip(segments, fixed)]
    transcript = Transcript(segments=_snap_all(segments, energy))
    _save(transcript_path, transcript)
    _record_pass_a_path(work, "per-track", fallback)
    last = parse_ts(transcript.segments[-1].start) if transcript.segments else 0.0
    by = {r: sum(s.speaker == r for s in transcript.segments) for r in TRACK_PROMPTS}
    print(f"[pass_a] {len(transcript.segments)} segments ({by['teacher']} teacher, "
          f"{by['student']} student), last at {fmt_ts(last)} of {fmt_ts(total)}"
          + (f"; window(s) {', '.join(map(str, fallback))} fell back to the mixed recording"
             if fallback else ""))
    return transcript


class TrackWindowDegraded(RuntimeError):
    """Every attempt at one track's window failed or was rejected: the window takes the mix."""


def _track_starts(tt: TrackTranscript) -> list[float]:
    return [t for t in (_start_or_inf(s.start) for s in tt.segments) if t != float("inf")]


def _transcribe_tracks_window(cfg: Config, get_client, role_windows: dict, i: int, tag: str,
                              span: str, tracks_dir: Path, energy, offset: float,
                              window_s: float) -> list[Segment]:
    """Both mic tracks of one window, in parallel, as speaker-labelled segments (window-relative).

    Each track is judged against how much that person actually spoke (quality.evaluate_track) and
    re-rolled warmer while it fails. A track that never passes raises TrackWindowDegraded, and the
    caller transcribes the window from the mix. Accepted transcripts are cached per track, so a
    retry only redoes the track that failed. Lines with no sound near the model's own start are
    dropped here, before snapping could move them onto nearby sound.
    """
    client = get_client()

    def judge(role: str, tt: TrackTranscript) -> TrackReport:
        return evaluate_track(_track_starts(tt), [x.text for x in tt.segments], energy[role],
                              FRAME_S, SOUND_DB, offset, window_s)

    def one(role: str) -> TrackTranscript:
        cache = tracks_dir / f"{role}_w{i:02d}.json"
        if cache.exists():
            return _load(cache, TrackTranscript)
        lo, hi = int(offset / FRAME_S), int((offset + window_s) / FRAME_S)
        if float((energy[role][lo:hi] > SOUND_DB).sum()) * FRAME_S < TRACK_SILENT_S:
            # Nobody spoke on this track here, and a model asked to transcribe silence invents.
            print(f"[pass_a] {tag} {role}: no speech on this track, nothing to transcribe", flush=True)
            result = TrackTranscript(segments=[])
            _save(cache, result)
            return result
        path, _ = role_windows[role][i - 1]
        problems: list[str] = []
        tries = max(1, cfg.window_attempts)
        for n in range(1, tries + 1):
            try:
                result = generate(
                    client, cfg.model, [upload_audio(client, path), TRACK_PROMPTS[role]],
                    TrackTranscript, temperature=0.2 + 0.2 * (n - 1),
                    thinking_level=cfg.pass_a_thinking,
                    debug_dump=tracks_dir / f"pass_a_partial_{role}_w{i:02d}.attempt{n}.json")
            except Exception as exc:  # noqa: BLE001 — a loop, a spent budget or a dead call: all re-roll
                problems.append(f"{type(exc).__name__}: {exc}")
                print(f"[pass_a] {tag} {role}: attempt {n} failed ({problems[-1]})", flush=True)
                continue
            report = judge(role, result)
            print(f"[pass_a] {tag} {role}: {report.summary()}", flush=True)
            if report.ok:
                _save(cache, result)
                return result
            _save(tracks_dir / f"{role}_w{i:02d}.attempt{n}.json", result)
            problems.append("; ".join(c.detail for c in report.failures))
            print(f"[pass_a] {tag} {role}: attempt {n} REJECTED: {problems[-1]}", flush=True)
        raise TrackWindowDegraded(f"{role} track, {tries} attempt(s): {problems[-1]}")

    print(f"[pass_a] {tag} {span}: transcribing the teacher and student tracks with {cfg.model}",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(TRACK_PROMPTS)) as pool:
        futures = {role: pool.submit(one, role) for role in TRACK_PROMPTS}
        results = {role: f.result() for role, f in futures.items()}
    segments = []
    dropped = 0
    for role, tt in results.items():
        for x in tt.segments:
            t = _start_or_inf(x.start)
            if t != float("inf") and is_ghost(energy[role], FRAME_S, SOUND_DB, offset + t):
                dropped += 1
                continue
            # Played audio on a mic (he played it out of his speakers) is not his speech; until
            # the transcript has a speaker value for it (ADR-0011, jld-8nv.4) it is marked in text.
            text = f"（再生音声）{x.text}" if x.played else x.text
            segments.append(Segment(start=x.start, speaker=role, text=text, uncertain=x.uncertain))
    segments.sort(key=lambda x: _start_or_inf(x.start))
    print(f"[pass_a] {tag}: {sum(x.speaker == 'teacher' for x in segments)} teacher + "
          f"{sum(x.speaker == 'student' for x in segments)} student segments"
          + (f"; dropped {dropped} line(s) with no sound on their own track" if dropped else ""),
          flush=True)
    return segments


def _start_or_inf(start: str) -> float:
    try:
        return parse_ts(start)
    except ValueError:
        return float("inf")


def _snap_all(segments: list[Segment], energy) -> list[Segment]:
    """Snap every start to its own speaker's track, then re-sort by the snapped time."""
    if energy is None:
        return segments
    timed = []
    for s in segments:
        t = parse_ts(s.start)
        db = energy.get(s.speaker)
        timed.append((snap_start(db, t) if db is not None else t, s))
    timed.sort(key=lambda pair: pair[0])
    return [s.model_copy(update={"start": fmt_ts(t)}) for t, s in timed]


def _pass_a(cfg: Config, work: Path, audio: Path, total: float, get_client) -> Transcript:
    """Transcribe the recording window by window and merge (see ADR-0005).

    One Gemini call per window, each cached as transcript_w<NN>.json so a failed
    window retries alone; the merged result is transcript.json, which is what every
    later stage reads.
    """
    transcript_path = work / "transcript.json"
    if transcript_path.exists():
        print(f"[pass_a] cached: {transcript_path}")
        return _load(transcript_path, Transcript)

    win_s = cfg.window_minutes * 60
    windows = split_windows(audio, work / "windows", win_s=win_s,
                            overlap_s=cfg.overlap_seconds, total=total)
    if len(windows) == 1:
        print(f"[pass_a] single pass over the whole recording ({fmt_ts(total)})")
    else:
        print(f"[pass_a] {len(windows)} windows of {cfg.window_minutes:g} min "
              f"with {cfg.overlap_seconds:g}s overlap")

    parts: list[list[Segment]] = []
    offsets: list[float] = []
    for i, (path, offset) in enumerate(windows, start=1):
        wpath = work / f"transcript_w{i:02d}.json"
        tag = f"window {i}/{len(windows)}"
        window_s = duration_seconds(path) if len(windows) > 1 else total
        if wpath.exists():
            print(f"[pass_a] {tag} cached: {wpath}")
            _report(tag, _load(wpath, Transcript).segments, window_s, cached=True)
        else:
            span = f"{fmt_ts(offset)}-{fmt_ts(min(total, offset + win_s) if win_s else total)}"
            wt = _transcribe_window(cfg, get_client, path, tag, span, window_s, wpath)
            _save(wpath, wt)
        parts.append(_load(wpath, Transcript).segments)
        offsets.append(offset)

    transcript = merge_windows(parts, offsets, win_s=win_s, total=total,
                               overlap_s=cfg.overlap_seconds)
    _save(transcript_path, transcript)
    _record_pass_a_path(work, "mixed")
    last = parse_ts(transcript.segments[-1].start) if transcript.segments else 0.0
    print(f"[pass_a] {len(transcript.segments)} segments, "
          f"last at {fmt_ts(last)} of {fmt_ts(total)}")
    # Diarization verdict over the merged hour (jld-86b). The per-window gates ask
    # whether speech went missing; this asks whether it landed on the right person,
    # and it only has a usable denominator once the windows are back together.
    print(f"[pass_a] {marker_verdict(transcript.segments)}")
    return transcript


def _transcribe_window(cfg: Config, get_client, audio: Path, tag: str, span: str,
                       window_s: float, wpath: Path) -> Transcript:
    """Transcribe one window, re-rolling it while the sanity gates say it is degraded.

    A degraded window is worth a retry precisely because it is one window: the call
    is minutes, not half an hour. Attempts are kept next to the window as
    transcript_w<NN>.attempt<N>.json, and if every attempt fails the least-bad one
    is promoted so the run can continue — loudly — rather than dying an hour in.
    """
    attempts: list[tuple[Transcript, WindowReport]] = []
    looped = 0
    tries = max(1, cfg.window_attempts)
    for n in range(1, tries + 1):
        client = get_client()
        # Structured output on disfluent audio loops; a warmer retry breaks the loop.
        temperature = 0.2 + 0.2 * (n - 1)
        note = "" if n == 1 else f" (attempt {n}, temperature {temperature:g})"
        print(f"[pass_a] {tag} {span}: uploading and transcribing with {cfg.model}{note} "
              "(dots = transcript streaming in)")
        try:
            wt: Transcript = generate(client, cfg.model,
                                      [upload_audio(client, audio), prompts.PASS_A],
                                      Transcript, temperature=temperature, progress=True,
                                      thinking_level=cfg.pass_a_thinking,
                                      debug_dump=wpath.with_name(
                                          f"pass_a_partial_{wpath.stem}.attempt{n}.json"))
        except RepetitionLoop as loop:
            # The abort is the whole point: a loop returns no usable JSON at all, so there
            # is no attempt to score and nothing to promote. The retry is the one already
            # here — a warmer re-roll is what breaks a degenerate decode.
            looped += 1
            print(f"[pass_a] {tag}: ABORTED — {loop}")
            if n < tries:
                print(f"[pass_a] {tag}: retrying this window at a higher temperature "
                      "(the other windows are unaffected)")
            continue
        report = _report(tag, wt.segments, window_s)
        attempts.append((wt, report))
        if report.ok:
            return wt
        if n < tries:
            _save(wpath.with_suffix(f".attempt{n}.json"), wt)
            print(f"[pass_a] {tag}: retrying this window (the other windows are unaffected)")

    if not attempts:
        raise RuntimeError(
            f"{tag}: every one of {looped} attempt(s) ended in a repetition loop, so this "
            f"window produced no transcript at all. Completed windows are cached, so "
            f"re-running resumes here. A shorter --window-minutes is the lever that helps: "
            f"the loop is the endpoint of a degradation curve that grows with how much "
            f"self-generated text is already in the response."
        )

    best, report = min(attempts, key=lambda pair: (len(pair[1].failures), -pair[1].coverage))
    print(f"[pass_a] {tag}: ALL {len(attempts)} attempts failed the sanity gates; "
          f"keeping the least-bad one ({report.summary()}). Delete {wpath.name} and re-run "
          "to try again, or accept it knowing this window is degraded.")
    return best


def _report(tag: str, segments: list[Segment], duration: float, cached: bool = False) -> WindowReport:
    """Print one stats line per window, plus whatever the gates have to say."""
    report = evaluate(segments, duration)
    print(f"[pass_a] {tag}: {report.summary()}")
    for check in report.failures:
        print(f"[pass_a] {tag}: FAILED {check.name} — {check.detail}")
    for check in report.warnings:
        print(f"[pass_a] {tag}: warning: {check.name} — {check.detail}")
    if cached and report.failures:
        print(f"[pass_a] {tag}: cached window is degraded; delete its file to re-transcribe it")
    return report


def merge_windows(parts: list[list[Segment]], offsets: list[float], win_s: float,
                  total: float, overlap_s: float = 30.0) -> Transcript:
    """Shift each window's timestamps to absolute time and stitch the windows together.

    Each overlap region is split at its midpoint: window N owns everything before it,
    window N+1 everything after, so no utterance is transcribed into the merged output
    twice and none is dropped. Segments a model placed past its own window are discarded
    rather than trusted — generously past the end for the final window, which has no
    successor to cover for it.
    """
    kept: list[tuple[float, Segment]] = []
    for i, (segments, offset) in enumerate(zip(parts, offsets, strict=True)):
        end = min(total, offset + win_s) if win_s else total
        lo = -1.0 if i == 0 else (offsets[i] + min(total, offsets[i - 1] + win_s)) / 2
        # The last window has no successor to own its tail, so it keeps everything up to
        # the end of the recording plus a drift allowance — models overshoot their own
        # span by tens of seconds (see the last-timestamp drift in any window). Past that
        # it is loop garbage, not lesson.
        hi = end + max(60.0, 0.1 * win_s) if i == len(parts) - 1 else (offsets[i + 1] + end) / 2
        for seg in segments:
            try:
                start = parse_ts(seg.start) + offset
            except ValueError:
                continue  # a malformed timestamp cannot be placed; drop it
            if lo <= start < hi:
                kept.append((start, seg))
    kept.sort(key=lambda pair: pair[0])
    return Transcript(segments=[
        seg.model_copy(update={"start": fmt_ts(start)})
        for start, seg in _drop_seam_duplicates(kept, overlap_s)
    ])


def _drop_seam_duplicates(kept: list[tuple[float, Segment]], overlap_s: float) -> list[tuple[float, Segment]]:
    """Drop a repeat of the same line by the same speaker near a seam.

    The midpoint rule already prevents duplicates, but two windows can time the same
    utterance a second or two apart and straddle the midpoint with it.
    """
    span = max(overlap_s, 5.0)
    out: list[tuple[float, Segment]] = []
    for start, seg in kept:
        key = _norm(seg.text)
        duplicate = False
        for prev, other in reversed(out):
            if start - prev > span:
                break
            if key and other.speaker == seg.speaker and _norm(other.text) == key:
                duplicate = True
                break
        if not duplicate:
            out.append((start, seg))
    return out


def _norm(text: str) -> str:
    return "".join(c for c in text if c not in " \t\u3000、。，．,.！？!?「」『』…")


def _near(notes: list[BoardNote], t0: float, t1: float) -> list[BoardNote]:
    """Board lines that belong with an exchange: typing trails speech, so look further after than before."""
    return [n for n in notes if t0 - BOARD_BEFORE <= n.t <= t1 + BOARD_AFTER]


def _relisten(client, model: str, clip: Path, cand: Candidate, thinking_level: str | None = None) -> Relisten:
    prompt = prompts.PASS_B.format(type=cand.type, rationale=cand.rationale, excerpt=cand.excerpt)
    return generate(client, model, [audio_part(clip), prompt], Relisten, thinking_level=thinking_level)


def _emit_transcript(cfg: Config, transcript: Transcript) -> None:
    lines = [f"# 授業の文字起こし {cfg.lesson_date}", "",
             f"Diarized transcript (verbatim — student errors preserved). Source: `{cfg.recording}`", ""]
    for seg in transcript.segments:
        name = "Soso先生" if seg.speaker == "teacher" else "Steven"
        flag = " ⚠️" if seg.uncertain else ""
        lines.append(f"- `[{seg.start}]` **{name}:**{flag} {seg.text}")
    path = cfg.out_dir / f"{cfg.date}-transcript.md"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[emit] {path}")
