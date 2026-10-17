"""jld-tlc spike: Pass A per Zoom participant track, against mixed-file Pass A (+ energy relabel).

Off the production path on purpose: its own work dir (work/spike-tlc/ by default), no archive,
no change to PASS_A, DEFAULT_MODEL or the thinking levels. Every stage caches its output; delete
a file to redo that stage.

Arms, all over the same window of the mix (work/<date>/windows/, same offsets):
  A0   mixed-file Pass A as it ran (work/<date>/transcript_wNN.json)
  A1   A0 with every clearly-contradicted label flipped by track energy (the 10/14 relabel rule)
  A2   teacher track first; Steven's track with the teacher's text as context; the share track
       transcribed as played audio over its active spans; merged by start, each start snapped
       to its own track's energy onset
  A2c  A2 with Steven's track transcribed WITHOUT the teacher context (normalisation control)

  uv run python scripts/spike_per_track.py transcribe --date 20261014 --window 3 --tracks <dir>
  uv run python scripts/spike_per_track.py compare    --date 20261014 --window 3 --tracks <dir>
  uv run python scripts/spike_per_track.py detect     --date 20261014 --window 3 --tracks <dir>
  uv run python scripts/spike_per_track.py kit        --date 20261014 --window 3 --tracks <dir>
  uv run python scripts/spike_per_track.py play       --date 20261014 --window 3
  uv run python scripts/spike_per_track.py score      --date 20261012,20261014 --window 3

Results and the decision they feed: decisions/0010-per-track-pass-a.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from jp_lesson_distill import prompts
from jp_lesson_distill.audio import clip_audio, duration_seconds, ffmpeg_exe, prep_audio, window_starts
from jp_lesson_distill.gemini import PASS_A_THINKING_LEVEL, DEFAULT_MODEL, audio_part, generate, make_client, upload_audio
from jp_lesson_distill.labeling import similarity
from jp_lesson_distill.models import CandidateList, Relisten, Segment, Transcript, fmt_ts, parse_ts
from jp_lesson_distill.quality import evaluate

WIN_S, OVERLAP_S = 20 * 60, 30.0
FRAME = 0.1          # seconds per energy frame
ACTIVE_DB = -45.0    # the 10/14 relabel rule (A1); kept as it was so A1 stays the arm it was
# Any sound at all. Zoom's suppression leaves exact digital silence (-120 dB), and Steven's soft
# はい／うーん／あー sit at -33 to -55 dB on his own track (measured on 10/14 w3) -- below the
# relabel rule's -45, so snapping, ends, ghosts and coverage use this instead.
SOUND_DB = -70.0
SNAP_S = 3.0         # how far a model start may sit from its track's energy onset
CLIP_PAD = 15.0

# Exact display-name prefixes (a loose glob matched digits inside the meeting id once).
ROLES = {"steven": "audioSteveSaito1", "teacher": "audio末武創一2", "share": "audio末武創一3"}


# --- single-speaker prompts (PASS_A stays untouched; these are spike-only) ---

TRACK_RULES = """Produce a complete transcript of THIS SPEAKER as JSON segments. Rules:
1. VERBATIM. Preserve errors exactly as spoken: wrong particles, wrong conjugations, wrong words, \
false starts, fillers (えーと、あの、あー…), self-corrections, trailing off. NEVER fix or smooth what \
was said. If something ungrammatical is said, transcribe the ungrammatical version.
2. If a word is mispronounced, render what you actually HEARD in kana, not the intended word.
3. Write Japanese in normal kana/kanji orthography; English as spoken. Mixed-language segments are fine.
4. One segment per utterance. Start a new segment whenever the speaker pauses for more than about \
one and a half seconds (that is usually where the other person spoke), and split utterances longer \
than about 30 seconds.
5. "start" is when the utterance begins, measured from the beginning of this recording, formatted \
"MM:SS".
6. Set "uncertain": true when you could not make the words out clearly (mumbling, garble). Do not \
guess-correct — write your best hearing and flag it.
7. Set "played": true for recorded audio played back (a textbook recording with other voices, a \
video) rather than this person speaking live.
8. The silences on this track are where the OTHER person was talking. Never write anything for a \
silence: if you hear nothing, there is no segment.

Transcribe the ENTIRE recording, start to finish."""

TRACK_TEACHER = f"""You are transcribing one participant's microphone track from a one-on-one Japanese \
lesson recorded over Zoom. Each participant was recorded to a separate file, so only ONE voice is on \
this track: the teacher, Soso先生, a native Japanese speaker who speaks mostly Japanese, occasionally \
English. The student's voice is on a different file; whenever the teacher is not speaking this track \
is silent.

{TRACK_RULES}"""

TRACK_STUDENT = f"""You are transcribing one participant's microphone track from a one-on-one Japanese \
lesson recorded over Zoom. Each participant was recorded to a separate file, so only ONE voice is on \
this track: the student, Steven, a native English speaker learning Japanese (intermediate), who \
speaks Japanese with learner mistakes and sometimes switches to English. The teacher's voice is on a \
different file; whenever the student is not speaking this track is silent.

His mistakes are the whole point of this transcript: it is used to find the errors he made.

{TRACK_RULES}"""

STUDENT_CONTEXT = """

CONTEXT. Below is what the teacher said during this same stretch of the lesson, transcribed from \
the teacher's own track, on the same clock as this recording. Use it only to follow what the student \
is answering and to recognise words the teacher said first. It must NOT change what you write for \
the student: when the teacher corrects or rephrases him, the student's line still contains the \
error exactly as he said it. Never copy the teacher's words into the student's lines unless you \
hear the student say them.

TEACHER (context only, not to be transcribed):
{context}"""

SHARE_PROMPT = """This clip is recorded audio that a Japanese teacher played to his student during an \
online lesson, captured from his shared computer sound — usually a textbook dialogue recording, \
sometimes a video. Transcribe it verbatim, one segment per line of dialogue (a new segment whenever \
the voice changes). "start" is measured from the beginning of this clip, "MM:SS". Set "uncertain" \
when you cannot make the words out. Set "played" to true. If the clip has no intelligible speech \
(music, a chime), return no segments."""


class TrackSegment(BaseModel):
    start: str = Field(description="Utterance start time from the beginning of the recording, 'MM:SS'")
    text: str
    uncertain: bool = Field(default=False, description="True if the words could not be made out clearly")
    played: bool = Field(default=False, description="True for recorded audio played back, not live speech")


class TrackTranscript(BaseModel):
    segments: list[TrackSegment]


# --- inputs ---

def find_tracks(tracks_dir: Path, mix_total: float) -> dict[str, Path]:
    """Role → file, by exact display-name prefix, refusing a partial or mismatched set."""
    found: dict[str, Path] = {}
    for role, prefix in ROLES.items():
        hits = sorted(p for p in tracks_dir.glob("*.m4a") if p.name.startswith(prefix))
        if len(hits) != 1:
            raise SystemExit(f"{role}: expected one file starting {prefix!r} in {tracks_dir}, got {hits}")
        d = duration_seconds(hits[0])
        if abs(d - mix_total) > 2.0:
            raise SystemExit(f"{role}: {hits[0].name} is {d:.1f}s, the mix is {mix_total:.1f}s")
        found[role] = hits[0]
    return found


def decode_db(path: Path) -> np.ndarray:
    raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    n = len(a) // 1600
    a = a[: n * 1600].reshape(n, 1600)
    return 10 * np.log10((a ** 2).mean(1) + 1e-12)


class Lesson:
    """Paths and cached inputs for one date/window of the spike."""

    def __init__(self, args):
        self.date, self.window = args.date, args.window
        self.prod = Path(args.prod_dir) / self.date
        self.root = Path(args.work_dir) / self.date
        self.out = self.root / f"w{self.window:02d}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.mix = self.prod / "audio.m4a"
        self.total = duration_seconds(self.mix)
        starts = window_starts(self.total, WIN_S, OVERLAP_S)
        self.offset = starts[self.window - 1]
        self.end = min(self.total, self.offset + WIN_S)
        self.tracks_dir = Path(args.tracks) if getattr(args, "tracks", None) else None

    def energy(self) -> dict[str, np.ndarray]:
        cache = self.root / "energy.npz"
        if not cache.exists():
            tracks = find_tracks(self.tracks_dir, self.total)
            arrays = {role: decode_db(p) for role, p in tracks.items()}
            arrays["mix"] = decode_db(self.mix)
            np.savez_compressed(cache, **arrays)
        return dict(np.load(cache))

    def track_window(self, role: str) -> Path:
        """The role's track, prepped like the mix and cut at the mix's window offset."""
        prepped = self.root / "tracks" / f"{role}.m4a"
        if not prepped.exists():
            prepped.parent.mkdir(parents=True, exist_ok=True)
            prep_audio(find_tracks(self.tracks_dir, self.total)[role], prepped)
        cut = self.root / "tracks" / f"{role}_w{self.window:02d}.m4a"
        if not cut.exists():
            clip_audio(prepped, cut, self.offset, self.end)
        return cut


# --- transcription ---

def _call(lesson: Lesson, name: str, fn) -> TrackTranscript:
    path = lesson.out / f"{name}.json"
    if path.exists():
        print(f"[{name}] cached: {path}")
        return TrackTranscript.model_validate_json(path.read_text())
    t0 = time.monotonic()
    result = fn()
    wall = time.monotonic() - t0
    path.write_text(result.model_dump_json(indent=2))
    cost = lesson.out / "cost.jsonl"
    with cost.open("a") as f:
        f.write(json.dumps({"call": name, "wall_s": round(wall, 1), "segments": len(result.segments)}) + "\n")
    print(f"[{name}] {len(result.segments)} segments in {wall:.0f}s")
    return result


def active_spans(db: np.ndarray, lo: float, hi: float, gap: float = 2.0, min_len: float = 0.5,
                 pad: float = 0.5) -> list[tuple[float, float]]:
    act = db > SOUND_DB
    spans: list[list[float]] = []
    for f in range(int(lo / FRAME), min(len(act), int(hi / FRAME))):
        if not act[f]:
            continue
        t = f * FRAME
        if spans and t - spans[-1][1] <= gap:
            spans[-1][1] = t + FRAME
        else:
            spans.append([t, t + FRAME])
    return [(max(lo, a - pad), min(hi, b + pad)) for a, b in spans if b - a >= min_len]


def transcribe(args) -> None:
    lesson = Lesson(args)
    energy = lesson.energy()
    client = make_client()
    level = PASS_A_THINKING_LEVEL

    def track_call(role: str, prompt: str):
        def fn():
            audio = lesson.track_window(role)
            print(f"[{role}] transcribing {audio.name} ({fmt_ts(lesson.offset)}-{fmt_ts(lesson.end)})")
            return generate(client, args.model, [upload_audio(client, audio), prompt], TrackTranscript,
                            progress=True, thinking_level=level)
        return fn

    teacher = _call(lesson, "teacher", track_call("teacher", TRACK_TEACHER))
    context = "\n".join(f"[{s.start}] {s.text}" for s in teacher.segments if not s.played)
    _call(lesson, "steven_ctx", track_call("steven", TRACK_STUDENT + STUDENT_CONTEXT.format(context=context)))
    _call(lesson, "steven_noctx", track_call("steven", TRACK_STUDENT))

    # Share track: only where it carries sound (2-5% of a lesson); timestamps are shifted to the window.
    def share():
        spans = active_spans(energy["share"], lesson.offset, lesson.end)
        print(f"[share] {len(spans)} active spans, {sum(b - a for a, b in spans):.0f}s")
        prepped = lesson.track_window("share").parent / "share.m4a"
        out: list[TrackSegment] = []
        for i, (a, b) in enumerate(spans):
            clip = lesson.out / "share_clips" / f"s{i:02d}.m4a"
            clip.parent.mkdir(exist_ok=True)
            if not clip.exists():
                clip_audio(prepped, clip, a, b)
            if b - a < 1.5:
                continue
            got = generate(client, args.model, [audio_part(clip), SHARE_PROMPT], TrackTranscript,
                           thinking_level=level)
            for s in got.segments:
                rel = a - lesson.offset + parse_ts(s.start)
                out.append(s.model_copy(update={"start": f"{int(rel) // 60:02d}:{rel % 60:04.1f}", "played": True}))
        return TrackTranscript(segments=out)
    _call(lesson, "share", share)


# --- merge: one clock, union sorted by start, each start snapped to its own track's energy ---

def snap(db: np.ndarray, s: float, lo: float, hi: float) -> tuple[float, bool]:
    """Nearest energy onset to model start `s`; `s` itself if there is none within SNAP_S."""
    act = db > SOUND_DB
    best = None
    for f in range(max(1, int((s - SNAP_S) / FRAME)), min(len(act), int((s + SNAP_S) / FRAME) + 1)):
        if act[f] and not act[f - 1]:
            if best is None or abs(f * FRAME - s) < abs(best - s):
                best = f * FRAME
    f = int(s / FRAME)
    if 0 <= f < len(act) and act[f]:  # inside speech: take the onset of this run if it is close
        g = f
        while g > 0 and act[g - 1]:
            g -= 1
        if s - g * FRAME <= SNAP_S:
            best = g * FRAME
    if best is None:
        return s, False
    return min(max(best, lo), hi), True


def merged(lesson: Lesson, steven_name: str) -> list[dict]:
    energy = lesson.energy()
    rows: list[dict] = []
    for name, role, db_key in (("teacher", "teacher", "teacher"), (steven_name, "student", "steven"),
                               ("share", "played", "share")):
        tt = TrackTranscript.model_validate_json((lesson.out / f"{name}.json").read_text())
        track_rows = []
        for s in tt.segments:
            raw = lesson.offset + parse_ts(s.start)
            start, snapped = snap(energy[db_key], raw, lesson.offset, lesson.end)
            track_rows.append({"start": start, "raw": raw, "snapped": snapped, "track": db_key,
                               "speaker": "played" if (s.played or role == "played") else role,
                               "text": s.text, "uncertain": s.uncertain})
        track_rows.sort(key=lambda r: r["start"])
        act = energy[db_key] > SOUND_DB
        for i, r in enumerate(track_rows):
            nxt = track_rows[i + 1]["start"] if i + 1 < len(track_rows) else lesson.end
            a, b = int(r["start"] / FRAME), max(int(r["start"] / FRAME) + 1, int(nxt / FRAME))
            on = np.nonzero(act[a:b])[0]
            # end = last active frame before this track's next segment; a span with no energy at
            # all on its own track is a line the model made up (no mic hears the other person).
            r["end"] = (a + on[-1] + 1) * FRAME if len(on) else r["start"]
            r["own_active"] = float(act[a:b].mean()) if b > a else 0.0
            # ghost: no sound on its own track anywhere near where the model put it
            near = energy[db_key][max(0, int((r["raw"] - 2) / FRAME)): int((r["raw"] + 3) / FRAME)]
            r["own_max_db"] = round(float(near.max()), 1) if len(near) else -120.0
        rows += track_rows
    rows.sort(key=lambda r: r["start"])
    return rows


def as_transcript(rows: list[dict], offset: float = 0.0) -> Transcript:
    """Rows → the production Segment schema. Played audio has no speaker value there, so it rides
    as a teacher line with a prefix detect can read."""
    return Transcript(segments=[Segment(
        start=f"{fmt_ts(r['start'] - offset)}", speaker="teacher" if r["speaker"] == "played" else r["speaker"],
        text=("（教科書の音声）" + r["text"]) if r["speaker"] == "played" else r["text"],
        uncertain=r["uncertain"]) for r in rows])


def a0_rows(lesson: Lesson) -> list[dict]:
    w = Transcript.model_validate_json((lesson.prod / f"transcript_w{lesson.window:02d}.json").read_text())
    rows = [{"start": lesson.offset + parse_ts(s.start), "speaker": s.speaker, "text": s.text,
             "uncertain": s.uncertain} for s in w.segments]
    rows.sort(key=lambda r: r["start"])
    for i, r in enumerate(rows):
        r["end"] = rows[i + 1]["start"] if i + 1 < len(rows) else min(lesson.end, r["start"] + 5)
    return rows


def energy_truth(energy, a: float, b: float) -> str | None:
    """The 10/14 rule: share > 50% = playback; else the track active > 30% and > 2x the other."""
    fa, fb = int(a / FRAME), int(b / FRAME)
    if fb <= fa:
        return None
    st, te, sh = ((energy[k][fa:fb] > ACTIVE_DB).mean() for k in ("steven", "teacher", "share"))
    if sh > 0.5:
        return "played"
    if st > 0.3 and st > 2 * te:
        return "student"
    if te > 0.3 and te > 2 * st:
        return "teacher"
    return None


def a1_rows(lesson: Lesson) -> list[dict]:
    energy = lesson.energy()
    rows = a0_rows(lesson)
    for r in rows:
        truth = energy_truth(energy, r["start"], r["end"])
        if truth in ("student", "teacher"):
            r["speaker"] = truth
    return rows


# --- comparison ---

FILLERS = re.compile(r"えーと|えっと|えー|あー|あの|うーん|んー|まあ")


def _cover(energy, rows, role_key: str, speaker: str, lesson: Lesson) -> float:
    """Share of a track's active frames inside spans labelled with its speaker (span = start→next start)."""
    act = energy[role_key] > SOUND_DB
    lo, hi = int(lesson.offset / FRAME), int(lesson.end / FRAME)
    covered = np.zeros(hi - lo, dtype=bool)
    starts = [r["start"] for r in rows] + [lesson.end]
    for i, r in enumerate(rows):
        if r["speaker"] == speaker:
            covered[max(0, int(starts[i] / FRAME) - lo): max(0, int(starts[i + 1] / FRAME) - lo)] = True
    return float((act[lo:hi] & covered).sum() / max(1, act[lo:hi].sum()))


def _onset_ok(energy, rows) -> str:
    """Segments whose labelled speaker's track has sound within -1..+2 s of the start.

    Checks timestamp and speaker together, from the tracks alone: a start placed 20 s early, or on
    the wrong person, lands where that voice is silent."""
    key = {"student": "steven", "teacher": "teacher", "played": "share"}
    ok = 0
    for r in rows:
        a, b = int((r["start"] - 1) / FRAME), int((r["start"] + 2) / FRAME)
        ok += bool((energy[key[r["speaker"]]][max(0, a):b] > SOUND_DB).any())
    return f"{ok}/{len(rows)}"


def stats(name: str, rows: list[dict], lesson: Lesson, energy) -> dict:
    t = as_transcript(rows, lesson.offset)
    rep = evaluate(t.segments, lesson.end - lesson.offset)
    spans = [(r["start"], (rows[i + 1]["start"] if i + 1 < len(rows) else lesson.end)) for i, r in enumerate(rows)]
    truths = [energy_truth(energy, a, b) for a, b in spans]
    graded = [(r, tr) for r, tr in zip(rows, truths) if tr is not None]
    agree = sum((r["speaker"] == tr) or (tr == "played" and r["speaker"] in ("played", "teacher"))
                for r, tr in graded)
    stu = [r for r in rows if r["speaker"] == "student"]
    stu_chars = sum(len(r["text"]) for r in stu)
    return {
        "arm": name, "segments": len(rows),
        "teacher": sum(r["speaker"] == "teacher" for r in rows), "student": len(stu),
        "played": sum(r["speaker"] == "played" for r in rows),
        "student_chars": stu_chars,
        "fillers_per_1000": round(1000 * sum(len(FILLERS.findall(r["text"])) for r in stu) / max(1, stu_chars), 1),
        "uncertain_student": sum(r["uncertain"] for r in stu),
        "ambiguous_by_energy": sum(tr is None for tr in truths),
        "speaker_agree_energy": f"{agree}/{len(graded)}",
        "start_on_own_voice": _onset_ok(energy, rows),
        "steven_speech_covered": round(_cover(energy, rows, "steven", "student", lesson), 3),
        "teacher_speech_covered": round(_cover(energy, rows, "teacher", "teacher", lesson), 3),
        "quality": rep.summary(), "quality_failures": [c.name for c in rep.failures],
    }


def compare(args) -> None:
    lesson = Lesson(args)
    energy = lesson.energy()
    lo, hi = int(lesson.offset / FRAME), int(lesson.end / FRAME)

    # Same clock? Mix activity vs the union of the tracks, at lags of up to ±2 s.
    mix = energy["mix"][lo:hi] > ACTIVE_DB
    union = np.zeros_like(mix)
    for k in ("steven", "teacher", "share"):
        union |= energy[k][lo:hi] > ACTIVE_DB
    lags = {lag: float((np.roll(union, lag) == mix).mean()) for lag in range(-20, 21)}
    best = max(lags, key=lags.get)
    only_mix = float((mix & ~union).sum() * FRAME)
    print(f"clock: mix vs tracks agree {lags[0]:.1%} at lag 0, best {lags[best]:.1%} at {best * FRAME:+.1f}s; "
          f"{only_mix:.0f}s of the window is sound in the mix and on no track")
    for k in ("steven", "teacher", "share"):
        seg = energy[k][lo:hi]
        silent = seg[seg <= ACTIVE_DB]
        quiet = ((seg > SOUND_DB) & (seg <= ACTIVE_DB)).mean()
        print(f"  {k}: active {(seg > ACTIVE_DB).mean():.1%}, quiet sound {quiet:.1%} "
              f"({SOUND_DB:g} to {ACTIVE_DB:g} dB), digital silence {(seg <= -100).mean():.1%}, "
              f"silence median {np.median(silent):.0f} dB")

    arms = {"A0": a0_rows(lesson), "A1": a1_rows(lesson),
            "A2": merged(lesson, "steven_ctx"), "A2c": merged(lesson, "steven_noctx")}
    table = [stats(k, v, lesson, energy) for k, v in arms.items()]
    for row in table:
        print(json.dumps(row, ensure_ascii=False))

    # Lines in a single-speaker transcript with no energy on their own track: the model made them up.
    for k in ("A2", "A2c"):
        ghosts = [r for r in arms[k] if r["own_max_db"] < SOUND_DB]
        unsnapped = [r for r in arms[k] if not r["snapped"]]
        print(f"{k}: {len(ghosts)} lines with no energy on their own track, {len(unsnapped)} starts not snapped")
        for r in ghosts:
            print(f"   {fmt_ts(r['start'])} {r['track']} ({r['own_max_db']:.0f} dB): {r['text'][:70]}")

    # A0's segments the energy rule could not decide: what does A2 have in the same span?
    a0 = arms["A0"]
    amb = [r for r in a0 if energy_truth(energy, r["start"], r["end"]) is None]
    clean = 0
    for r in amb:
        inside = [x for x in arms["A2"] if r["start"] - 0.5 <= x["start"] < r["end"]]
        if len({x["speaker"] for x in inside}) >= 2:
            clean += 1
    print(f"A0 ambiguous segments: {len(amb)}; A2 splits {clean} of them into turns from 2+ tracks")

    drift = timestamp_drift(arms["A0"], arms["A2"])
    print("A0 start minus A2 (energy-snapped) start for the same line, by 5 minutes of the window:")
    for m, ds in drift.items():
        print(f"   {m}: n={len(ds):2d} median {np.median(ds):+5.1f}s  max |d| {np.max(np.abs(ds)):4.1f}s")

    (lesson.out / "compare.json").write_text(json.dumps(
        {"stats": table, "drift": {k: [round(x, 1) for x in v] for k, v in drift.items()}},
        ensure_ascii=False, indent=1))
    for k, rows in arms.items():
        (lesson.out / f"merged_{k}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        lines = [f"- `[{fmt_ts(r['start'])}]` **{r['speaker']}:**{' ⚠️' if r['uncertain'] else ''} {r['text']}"
                 for r in rows]
        (lesson.out / f"transcript_{k}.md").write_text("\n".join(lines) + "\n")
    side_by_side(lesson, arms)


def timestamp_drift(a0: list[dict], a2: list[dict]) -> dict[str, list[float]]:
    """How far the mixed-file start sits from the energy-snapped one, for lines both arms have.

    A line pairs with the A0 segment within ±60 s whose text BEGINS like it; only snapped A2
    lines of 8+ characters with a confident match count.
    """
    out: dict[str, list[float]] = {}
    base = min(r["start"] for r in a0)
    for r in a2:
        if not r.get("snapped") or len(r["text"]) < 8 or r["speaker"] == "played":
            continue
        best, score = None, 0.0
        for c in a0:
            if abs(c["start"] - r["start"]) > 60:
                continue
            # against the START of the A0 text, so a line in the middle of a merged turn is not
            # paired with that turn's start
            s = similarity(r["text"], c["text"][: len(r["text"]) + 4])
            if s > score:
                best, score = c, s
        if best is None or score < 0.6:
            continue
        m = int((r["start"] - base) // 300) * 5
        out.setdefault(f"{m:02d}-{m + 5:02d}min", []).append(best["start"] - r["start"])
    return dict(sorted(out.items()))


def _student_text(rows: list[dict], a: float, b: float) -> str:
    """Student text whose span overlaps [a, b): a long turn that began earlier still counts."""
    return " / ".join(r["text"] for r in rows
                      if r["speaker"] == "student" and r["start"] < b and max(r["end"], r["start"] + 1) > a)


def side_by_side(lesson: Lesson, arms) -> None:
    """Each A2 student line beside what A1 and A2c have for Steven over the same span."""
    out = ["| t | A2 (ctx) | A2c (no ctx) | A1 (mixed, relabelled) | sim A2/A1 |", "|---|---|---|---|---|"]
    for r in arms["A2"]:
        if r["speaker"] != "student":
            continue
        a, b = r["start"], max(r["end"], r["start"] + 1)
        c = _student_text(arms["A2c"], a, b)
        m = _student_text(arms["A1"], a - 3, b)
        out.append(f"| {fmt_ts(a)} | {r['text']} | {c} | {m} | {similarity(r['text'], m):.2f} |")
    (lesson.out / "student_side_by_side.md").write_text("\n".join(out) + "\n")


# --- detect + Pass B on the window, same prompts as production ---

def detect(args) -> None:
    lesson = Lesson(args)
    client = make_client()
    arm_names = args.arms.split(",")
    for arm in arm_names:
        rows = json.loads((lesson.out / f"merged_{arm}.json").read_text())
        t = as_transcript(rows)  # absolute timestamps so candidates can be clipped from the mix
        cpath = lesson.out / f"candidates_{arm}.json"
        if not cpath.exists():
            print(f"[detect] {arm}")
            cands = generate(client, args.model, [prompts.DETECT.format(transcript=t.model_dump_json())],
                             CandidateList, progress=True)
            cpath.write_text(cands.model_dump_json(indent=2))
        cands = CandidateList.model_validate_json(cpath.read_text())
        rdir = lesson.out / f"relisten_{arm}"
        rdir.mkdir(exist_ok=True)
        kept = []
        for i, c in enumerate(sorted(cands.candidates, key=lambda c: c.priority, reverse=True), start=1):
            saved = rdir / f"m{i:02d}.json"
            if not saved.exists():
                t0, t1 = parse_ts(c.t_start), parse_ts(c.t_end)
                clip = rdir / f"m{i:02d}.m4a"
                clip_audio(lesson.mix, clip, t0 - CLIP_PAD, min(lesson.total, t1 + CLIP_PAD))
                prompt = prompts.PASS_B.format(type=c.type, rationale=c.rationale, excerpt=c.excerpt)
                r = generate(client, args.model, [audio_part(clip), prompt], Relisten)
                saved.write_text(r.model_dump_json(indent=2))
            r = Relisten.model_validate_json(saved.read_text())
            if r.keep:
                kept.append({"t": c.t_start, "type": r.type, "student_verbatim": r.student_verbatim,
                             "teacher_correction": r.teacher_correction, "confidence": r.confidence})
        (lesson.out / f"moments_{arm}.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1))
        print(f"[detect] {arm}: {len(cands.candidates)} candidates, {len(kept)} moments kept")


# --- verbatim listening kit: Steven's track, three blinded texts ---

class KitItem(BaseModel):
    id: str
    t: float
    clip: str
    order: list[str]           # arm names in the order shown as 1/2/3
    texts: dict[str, str]
    answer: Optional[str] = None  # the arm judged closest, "tie", or "none"
    note: str = ""


def _trim(text: str, ref: str, slack: int = 3) -> str:
    """The stretch of `text` that lines up with `ref`, so a merged block is judged on the same words.

    A1's segments are often whole merged turns, and its starts drift by up to ~25 s, so the
    caller pulls generously and this cuts back to the first..last run of 2+ characters shared
    with the reference line (plus a little slack, so a dropped or added filler stays visible)."""
    import difflib
    blocks = [m for m in difflib.SequenceMatcher(None, text, ref, autojunk=False).get_matching_blocks()
              if m.size >= 2]
    if not blocks:
        return ""
    lo, hi = max(0, blocks[0].a - slack), min(len(text), blocks[-1].a + blocks[-1].size + slack)
    seps = " 、。，,？?！!/"
    while lo > 0 and text[lo - 1] not in seps:  # never cut inside a word
        lo -= 1
    while hi < len(text) and text[hi - 1] not in seps:
        hi += 1
    return text[lo:hi].strip(" /")


def _same(a: str, b: str) -> bool:
    """Equal once spelling choices that say nothing about what he said are set aside."""
    def norm(t: str) -> str:
        t = re.sub(r"(?i)\bok\b", "オッケー", t)
        return "".join(c for c in t if c not in " 　、。，．,.！？!?「」『』…/")
    return norm(a) == norm(b)


def kit(args) -> None:
    lesson = Lesson(args)
    arms = {k: json.loads((lesson.out / f"merged_{k}.json").read_text()) for k in ("A1", "A2", "A2c")}
    steven = lesson.root / "tracks" / "steven.m4a"
    rng = random.Random(args.seed)
    items = []
    for r in arms["A2"]:
        if r["speaker"] != "student" or len(r["text"]) < 6:
            continue
        a, b = r["start"], max(r["end"], r["start"] + 1)
        texts = {"A2": r["text"],
                 "A2c": _trim(_student_text(arms["A2c"], a - 2, b + 2), r["text"]),
                 "A1": _trim(_student_text(arms["A1"], a - 25, b + 5), r["text"])}
        if all(_same(texts["A2"], v) for v in texts.values() if v) and (texts["A1"] or _same(texts["A2"], texts["A2c"])):
            continue  # nothing to judge: every arm with text agrees (a line A1 lost still counts
            # when the two per-track arms disagree about it)
        items.append((a, b, texts))
    rng.shuffle(items)
    items = sorted(items[: args.n], key=lambda x: x[0])
    kdir = lesson.out / "verbatim_kit"
    (kdir / "clips").mkdir(parents=True, exist_ok=True)
    out = []
    for n, (a, b, texts) in enumerate(items, start=1):
        clip = f"clips/v{n:02d}.m4a"
        if not (kdir / clip).exists():
            clip_audio(steven, kdir / clip, a - 0.5, b + 0.7)
        order = list(texts)
        rng.shuffle(order)
        out.append(KitItem(id=f"v{n:02d}", t=a, clip=clip, order=order, texts=texts))
    (kdir / "items.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in out))
    print(f"[kit] {len(out)} clips in {kdir}")


def play(args) -> None:
    kdir = Path(args.work_dir) / args.date / f"w{args.window:02d}" / "verbatim_kit"
    path = kdir / "items.jsonl"
    items = [KitItem.model_validate_json(x) for x in path.read_text().splitlines()]
    save = lambda: path.write_text("".join(i.model_dump_json() + "\n" for i in items))  # noqa: E731
    print("Which text is closest to what Steven actually said, errors and all? The clip is from his\n"
          "own track, so only his voice is on it. Judge the words you hear; a text that runs on a\n"
          "word or two past the clip is fine, a missing or invented word, filler or error is not.\n"
          "  [1/2/3] that one   [t] tie   [n] none of them   [r] replay   [q] save and quit\n"
          "  Optionally add a note after the key, e.g. '2 drops the えーと'.")
    # Shuffled, so a session stopped halfway is still a fair sample of the window.
    for item in sorted(items, key=lambda i: random.Random(f"{args.seed}{i.id}").random()):
        if item.answer:
            continue
        print(f"\n[{item.id}] {fmt_ts(item.t)}")
        for k, arm in enumerate(item.order, start=1):
            print(f"  {k}. {item.texts[arm] or '(nothing)'}")
        while True:
            subprocess.run(["afplay", str(kdir / item.clip)], check=False)
            ans = input("  closest? ").strip()
            key, _, note = ans.partition(" ")
            if key == "r":
                continue
            if key == "q":
                save()
                return
            if key in ("1", "2", "3"):
                item.answer = item.order[int(key) - 1]
            elif key in ("t", "n"):
                item.answer = {"t": "tie", "n": "none"}[key]
            else:
                continue
            item.note = note
            save()
            break
    done = [i for i in items if i.answer]
    from collections import Counter
    print(f"\n{len(done)}/{len(items)} answered: {dict(Counter(i.answer for i in done))}")


def score(args) -> None:
    """Tally the verbatim kit answers for every date given (comma-separated --date)."""
    from collections import Counter
    total: Counter = Counter()
    for date in args.date.split(","):
        path = Path(args.work_dir) / date / f"w{args.window:02d}" / "verbatim_kit" / "items.jsonl"
        items = [KitItem.model_validate_json(x) for x in path.read_text().splitlines()]
        c = Counter(i.answer for i in items if i.answer)
        total += c
        print(f"{date}: {sum(c.values())}/{len(items)} answered {dict(c)}")
        for i in items:
            if i.answer and i.note:
                print(f"   {i.id} {fmt_ts(i.t)} -> {i.answer}: {i.note}")
    print(f"all: {dict(total)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["transcribe", "compare", "detect", "kit", "play", "score"])
    p.add_argument("--date", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--tracks")
    p.add_argument("--work-dir", default="work/spike-tlc")
    p.add_argument("--prod-dir", default="work")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--arms", default="A1,A2,A2c")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=14)
    args = p.parse_args()
    {"transcribe": transcribe, "compare": compare, "detect": detect, "kit": kit, "play": play, "score": score}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
