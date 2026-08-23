"""Stage orchestration. Each stage caches its output under work/<date>/ and is
skipped when the output file already exists — delete a stage file to redo it."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .audio import clip_audio, duration_seconds, prep_audio, split_windows
from .gemini import audio_part, generate, make_client, upload_audio
from .models import (
    Candidate,
    CandidateList,
    Moment,
    MomentsFile,
    Relisten,
    Segment,
    Transcript,
    fmt_ts,
    parse_ts,
)
from .quality import WindowReport, evaluate

CLIP_PAD = 15.0  # seconds of context on each side of a candidate


@dataclass
class Config:
    recording: Path
    date: str  # YYYYMMDD
    model: str = "gemini-pro-latest"
    work_dir: Path = Path("work")
    out_dir: Path = Path("/Users/stevensaito/Docs/Vault-GeneralNotes/_raw")
    skip_pass_b: bool = False
    max_moments: int = 40
    window_minutes: float = 20.0  # 0 = transcribe the whole recording in one call
    overlap_seconds: float = 30.0
    window_attempts: int = 2  # re-rolls of a window the sanity gates reject

    @property
    def lesson_date(self) -> str:
        return f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"


def _load(path: Path, schema):
    return schema.model_validate_json(path.read_text())


def _save(path: Path, obj) -> None:
    path.write_text(obj.model_dump_json(indent=2))


def run(cfg: Config) -> Path | None:
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

    # pass A
    def get_client():
        nonlocal client
        client = client or make_client()
        return client

    transcript = _pass_a(cfg, work, audio, total, get_client)

    # detect
    candidates_path = work / "candidates.json"
    if candidates_path.exists():
        print(f"[detect] cached: {candidates_path}")
    else:
        client = client or make_client()
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
        client = client or make_client()
        clips_dir = work / "clips"
        clips_dir.mkdir(exist_ok=True)
        moments: list[Moment] = []
        for i, cand in enumerate(ordered, start=1):
            t0, t1 = parse_ts(cand.t_start), parse_ts(cand.t_end)
            clip = clips_dir / f"m{i:02d}.m4a"
            if not clip.exists():
                clip_audio(audio, clip, t0 - CLIP_PAD, min(total, t1 + CLIP_PAD))
            print(f"[pass_b] {i}/{len(ordered)} re-listening {cand.t_start}-{cand.t_end} ({cand.type})")
            r: Relisten = _relisten(client, cfg.model, clip, cand)
            if not r.keep:
                print(f"[pass_b]   rejected: {r.explanation}")
                continue
            moments.append(Moment(
                id=f"m{i:02d}", t_start=t0, t_end=t1, type=r.type,
                student_verbatim=r.student_verbatim, teacher_correction=r.teacher_correction,
                explanation=r.explanation, confidence=r.confidence,
            ))
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
    return dest


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
    last = parse_ts(transcript.segments[-1].start) if transcript.segments else 0.0
    print(f"[pass_a] {len(transcript.segments)} segments, "
          f"last at {fmt_ts(last)} of {fmt_ts(total)}")
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
    for n in range(1, max(1, cfg.window_attempts) + 1):
        client = get_client()
        # Structured output on disfluent audio loops; a warmer retry breaks the loop.
        temperature = 0.2 + 0.2 * (n - 1)
        note = "" if n == 1 else f" (attempt {n}, temperature {temperature:g})"
        print(f"[pass_a] {tag} {span}: uploading and transcribing with {cfg.model}{note} "
              "(dots = transcript streaming in)")
        wt: Transcript = generate(client, cfg.model, [upload_audio(client, audio), prompts.PASS_A],
                                  Transcript, temperature=temperature, progress=True)
        report = _report(tag, wt.segments, window_s)
        attempts.append((wt, report))
        if report.ok:
            return wt
        if n < max(1, cfg.window_attempts):
            _save(wpath.with_suffix(f".attempt{n}.json"), wt)
            print(f"[pass_a] {tag}: retrying this window (the other windows are unaffected)")

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


def _relisten(client, model: str, clip: Path, cand: Candidate) -> Relisten:
    prompt = prompts.PASS_B.format(type=cand.type, rationale=cand.rationale, excerpt=cand.excerpt)
    return generate(client, model, [audio_part(clip), prompt], Relisten)


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
