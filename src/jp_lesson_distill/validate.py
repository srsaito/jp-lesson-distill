"""`distill label` and `distill score` — the diarization ground-truth loop (jld-dli).

Everything here is offline once the clips are cut: no API, no network. Answers are
written after every keystroke, so a session survives being interrupted and resumes
where it stopped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .audio import clip_audio, duration_seconds
from .labeling import (
    GRADED,
    Kit,
    build_items,
    find_script,
    mcnemar,
    parse_script,
    score,
)
from .models import Transcript, fmt_ts

PROMPT = """
  [t] Soso先生                     [s] Steven
  [p] textbook audio playing       [o] someone else in the room
  [?] cannot tell
  [r] replay      [b] back      [q] save and quit

  Judge the VOICE, not the words. When the two of them act out a 会話, the voice is
  still [t] or [s] even though the line belongs to 利用者 or 司書. When Soso先生 plays
  the textbook recording, those actors are [p] — nobody in the room said it.
"""


def _load_segments(path: Path):
    return Transcript.model_validate_json(path.read_text()).segments


def build(recording: Path, date: str, work_dir: Path, compare: Path | None,
          n_random: int, n_contested: int, n_scripted: int, seed: int,
          script_path: Path | None = None) -> Path:
    work = work_dir / date
    transcript_path = work / "transcript.json"
    audio = work / "audio.m4a"
    if not transcript_path.exists():
        raise SystemExit(f"no transcript to sample from: {transcript_path} (run `distill run` first)")
    if not audio.exists():
        raise SystemExit(f"no prepped audio: {audio}")

    segments = _load_segments(transcript_path)
    other = _load_segments(compare) if compare else None
    script_path = script_path or (find_script(recording) if recording else None)
    script = parse_script(script_path.read_text()) if script_path else None
    if script_path:
        print(f"[label] textbook dialogue: {script_path.name} ({len(script or [])} lines) — "
              "a lesson reviewing a dialogue this file does not cover will have a thin "
              "scripted stratum; pass --script to point at another one")
    else:
        print("[label] no Z_kaiwa_scripts_*.md next to the recording — "
              "skipping the scripted stratum")

    items = build_items(segments, compare=other, script=script, n_random=n_random,
                        n_contested=n_contested, n_scripted=n_scripted, seed=seed)
    if not items:
        raise SystemExit("nothing to sample")

    out = work / "labeling"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    total = duration_seconds(audio)
    for item in items:
        clip = out / item.clip
        if not clip.exists():
            start = max(0.0, item.t_start - 2.0)
            clip_audio(audio, clip, start, min(total, item.t_start + 12.0))

    kit = Kit(date=date, items=items)
    kit_path = out / "items.jsonl"
    if kit_path.exists():
        existing = {i.id: i.truth for i in Kit.load(kit_path, date).items if i.truth}
        for item in kit.items:
            item.truth = existing.get(item.id)
        print(f"[label] kept {sum(1 for i in kit.items if i.truth)} answers from the existing kit")
    kit.save(kit_path)

    counts: dict[str, int] = {}
    for item in items:
        counts[item.stratum] = counts.get(item.stratum, 0) + 1
    print(f"[label] {len(items)} clips in {out}: "
          + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return kit_path


def play(date: str, work_dir: Path) -> None:
    """Listen and label. The model's guess is never shown until scoring."""
    kit_path = work_dir / date / "labeling" / "items.jsonl"
    if not kit_path.exists():
        raise SystemExit(f"no kit at {kit_path} — run `distill label` first")
    kit = Kit.load(kit_path, date)
    root = kit_path.parent
    queue = sorted(kit.items, key=lambda i: i.order)
    todo = [i for i in queue if not i.truth]
    print(f"[label] {len(todo)} of {len(queue)} clips left.")
    print(PROMPT)

    keys = {"t": "teacher", "s": "student", "p": "played", "o": "other", "?": "unsure"}
    pos = 0
    while 0 <= pos < len(todo):
        item = todo[pos]
        print(f"\n[{pos + 1}/{len(todo)}] {fmt_ts(item.t_start)}  ({item.stratum})")
        print(f"    {item.text}")
        while True:
            _afplay(root / item.clip)
            answer = input("    who spoke? ").strip().lower()
            if answer == "r":
                continue
            if answer == "q":
                kit.save(kit_path)
                print(f"[label] saved {len(kit.labelled)}/{len(kit.items)} to {kit_path}")
                return
            if answer == "b":
                pos = max(0, pos - 1)
                break
            if answer in keys:
                item.truth = keys[answer]
                kit.save(kit_path)  # after every answer: a session can die at any moment
                pos += 1
                break
            print("    t / s / p / o / ? / r / b / q")
    kit.save(kit_path)
    print(f"\n[label] done — {len(kit.labelled)}/{len(kit.items)} labelled in {kit_path}")


def _afplay(clip: Path) -> None:
    try:
        subprocess.run(["afplay", str(clip)], check=False)
    except FileNotFoundError:  # not macOS
        print(f"    (no afplay; the clip is at {clip})")


def report(date: str, work_dir: Path, against: Path | None) -> None:
    kit_path = work_dir / date / "labeling" / "items.jsonl"
    if not kit_path.exists():
        raise SystemExit(f"no kit at {kit_path}")
    kit = Kit.load(kit_path, date)
    graded = [i for i in kit.items if i.truth in GRADED]
    unsure = [i for i in kit.items if i.truth == "unsure"]
    if not graded:
        raise SystemExit("nothing labelled yet — run `distill label --play`")

    print(f"\nground truth: {len(graded)} clips labelled"
          + (f", {len(unsure)} marked unclear (excluded)" if unsure else ""))

    scores, confusion = score(kit.items)
    print("\nPass A accuracy against what you heard:")
    for s in scores:
        print(f"  {s.stratum:12s} {s.correct:3d}/{s.n:<3d}  {s.accuracy:5.0%}")
    if confusion:
        print("  errors by direction:")
        for direction, n in sorted(confusion.items(), key=lambda kv: -kv[1]):
            print(f"    {direction:24s} {n}")
        print("    (truth student -> labelled teacher is the expensive one: detect reads it "
              "as native speech and never flags the error)")
    nonparticipant = [i for i in graded if i.truth in ("played", "other")]
    if nonparticipant:
        print(f"  {len(nonparticipant)} of {len(graded)} clips are not either participant "
              f"({sum(1 for i in nonparticipant if i.truth == 'played')} textbook audio, "
              f"{sum(1 for i in nonparticipant if i.truth == 'other')} someone else) — "
              "Pass A cannot label these correctly at all, see jld-lg6")

    if against:
        repaired = {}
        for seg in _load_segments(against):
            repaired[seg.start] = seg.speaker
        after = {i.id: repaired.get(fmt_ts(i.t_start), i.speaker) for i in kit.items}
        rscores, rconfusion = score(kit.items, labels=after)
        print(f"\n{against.name} accuracy on the same clips:")
        for s in rscores:
            print(f"  {s.stratum:12s} {s.correct:3d}/{s.n:<3d}  {s.accuracy:5.0%}")
        fixed, broken, p = mcnemar(kit.items, {}, after)
        print(f"\npaired comparison (McNemar): fixed {fixed}, broke {broken}, p = {p:.4f}")
        if broken:
            print("  a repair that breaks lines that were already right is a bad trade — "
                  "check those before shipping it")


def dump(date: str, work_dir: Path) -> None:
    """Write the labelled kit somewhere durable, for reuse as a fixture."""
    kit_path = work_dir / date / "labeling" / "items.jsonl"
    kit = Kit.load(kit_path, date)
    payload = [
        {"t_start": i.t_start, "text": i.text, "truth": i.truth, "stratum": i.stratum}
        for i in kit.items if i.truth
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
