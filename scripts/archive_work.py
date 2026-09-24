"""Copy each lesson's irreplaceable pipeline outputs to OneDrive, next to its recording.

`work/` is gitignored and lives on one laptop, but part of it cannot be regenerated: Gemini
output is nondeterministic and costs money (a re-run is a different transcript, not the same
one again), and the blind diarization labels are hours of listening. Audio can be re-cut
from the recording at any time, so it is left out — that is also most of work/'s size.

Layout — each lesson folder gets one archive folder, and inside it one folder per lesson
date that mirrors work/ exactly:

    日本語/Soso/Vol 3/L13/distill-archive/20260817/20260817/transcript.json
                                                  hc9-2-verify/20260817/labeling/items.jsonl
                                                  ...

so restoring a lesson is a plain copy back over work/ (see docs/archive.md). Re-running is
safe: a file is copied only when it is missing from the archive or newer in work/.

Usage:  uv run python scripts/archive_work.py [--dry-run] [--date YYYYMMDD ...]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

SOSO = Path.home() / "Library/CloudStorage/OneDrive-個人用/日本語/Soso"
ARCHIVE_DIR = "distill-archive"
AUDIO = {".m4a", ".mp3", ".wav", ".aiff", ".mp4", ".mov"}

# Lesson outputs that live outside work/<date>/. Each file is attributed to the lesson date
# in its path; the hc91 thinking-level runs are all window 1 of 2026-08-17 (ADR-0007).
EXTRA_ROOTS = {"hc9-2-verify": None, "ref-windowed": None, "hc91-window": "20260817"}

README = """\
# distill-archive

Pipeline outputs from https://github.com/srsaito/jp-lesson-distill for the recordings in
this folder, one subfolder per lesson date. Written by `scripts/archive_work.py`; the repo's
`docs/archive.md` explains it.

Each date folder mirrors the repo's gitignored `work/` directory, so to restore a lesson:

    rsync -a "<this folder>/<YYYYMMDD>/" ~/Dev/jp-lesson-distill/work/

What is here: Gemini output (Pass A transcripts per window, candidates, moments, the Gemini
3.5 Transcribe comparison responses, failed attempts and partials) and the blind diarization
labels (`labeling/items.jsonl`). None of it can be regenerated identically.

What is NOT here: audio. Everything audio-shaped in work/ is re-cut from the recording.

This folder holds what was said in the lessons — keep it private. The repo is public and
only ever carries text-free summaries (tests/fixtures/).
"""


def main_work_dir() -> Path:
    """The main checkout's work/, even when this runs from a linked worktree.

    Worktrees share one .git but each has its own (usually empty) work/; the lesson data
    lives only in the main checkout's, next to the shared .git directory.
    """
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent / "work"


def recording_folders() -> dict[str, Path]:
    """Lesson date -> the OneDrive folder holding that lesson's recording."""
    found: dict[str, Path] = {}
    files = [f for f in SOSO.rglob("*") if ARCHIVE_DIR not in f.parts and "Z_audio" not in str(f)]
    for f in files:
        m = re.match(r"(?i)soso_(\d{8})_", f.name)
        if m and f.suffix.lower() in AUDIO:
            found.setdefault(m.group(1), f.parent)
    # A few early lessons have no recording in OneDrive, only the vault outputs
    # (20260312-moments.json) filed in the lesson folder. That still names the folder.
    for f in files:
        m = re.match(r"(\d{8})-(moments|transcript)\.", f.name)
        if m:
            found.setdefault(m.group(1), f.parent)
    return found


def keepable(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() not in AUDIO
            and "__pycache__" not in p.parts and p.name != ".DS_Store"]


def plan(work: Path) -> dict[str, list[Path]]:
    by_date: dict[str, list[Path]] = {}
    for d in sorted(work.iterdir()):
        if re.fullmatch(r"20\d{6}", d.name):  # 1999xxxx are synthetic smoke-test lessons
            by_date.setdefault(d.name, []).extend(keepable(d))
    for name, fixed_date in EXTRA_ROOTS.items():
        root = work / name
        if not root.exists():
            continue
        for p in keepable(root):
            m = re.search(r"/(20\d{6})/", str(p))
            date = fixed_date or (m.group(1) if m else None)
            if date:
                by_date.setdefault(date, []).append(p)
    return by_date


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="say what would be copied, copy nothing")
    ap.add_argument("--date", nargs="*", help="only these lesson dates (default: all)")
    ap.add_argument("--work", type=Path, help="work/ to archive (default: the main checkout's)")
    args = ap.parse_args()

    work = args.work or main_work_dir()
    folders = recording_folders()
    copied = skipped = 0
    for date, files in sorted(plan(work).items()):
        if args.date and date not in args.date:
            continue
        folder = folders.get(date)
        if folder is None:
            print(f"{date}: no recording found under {SOSO} — NOT archived")
            continue
        dest_root = folder / ARCHIVE_DIR / date
        new = 0
        for src in files:
            dest = dest_root / src.relative_to(work)
            if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
                skipped += 1
                continue
            new += 1
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        copied += new
        print(f"{date}: {new:3d} new of {len(files):3d} files -> "
              f"{dest_root.relative_to(SOSO.parent)}")
        if not args.dry_run:
            (folder / ARCHIVE_DIR / "README.md").write_text(README)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied} files; {skipped} already archived and up to date")


if __name__ == "__main__":
    main()
