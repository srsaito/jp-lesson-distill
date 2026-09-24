"""Copy each lesson's irreplaceable outputs to OneDrive, next to its recording (docs/archive.md).

`work/` is gitignored and lives on one laptop, but part of it cannot be regenerated: Gemini
output is nondeterministic and billed (a re-run is a different transcript, not the same one
again), and blind diarization labels are hours of listening. Audio is left out — everything
audio-shaped in work/ is re-cut from the recording, and it is most of work/'s size.

Each lesson folder in OneDrive gets one archive folder, and inside it one folder per lesson
date that mirrors work/ exactly, so restoring a lesson is a plain copy back over work/:

    日本語/Soso/Vol 3/L13/distill-archive/20260817/20260817/transcript.json
                                                  hc9-2-verify/20260817/labeling/items.jsonl

Runs automatically at the end of every `distill run` and after every listening session;
`distill archive` does it by hand. A file is copied only when it is missing from the archive
or newer in work/, so it is always safe to repeat.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SOSO = Path.home() / "Library/CloudStorage/OneDrive-個人用/日本語/Soso"
ARCHIVE_DIR = "distill-archive"
AUDIO = {".m4a", ".mp3", ".wav", ".aiff", ".mp4", ".mov"}
LESSON_DATE = re.compile(r"20\d{6}")  # 1999xxxx are synthetic smoke-test lessons

# Folders whose paths carry no lesson date. The thinking-level runs are all window 1 of
# 2026-08-17 (ADR-0007).
UNDATED = {"hc91-window": "20260817"}

README = """\
# distill-archive

Pipeline outputs from https://github.com/srsaito/jp-lesson-distill for the recordings in
this folder, one subfolder per lesson date. Written automatically at the end of every
`distill run`; the repo's `docs/archive.md` explains it.

Each date folder mirrors the repo's gitignored `work/` directory, so to restore a lesson:

    rsync -a "<this folder>/<YYYYMMDD>/" ~/Dev/jp-lesson-distill/work/

What is here: Gemini output (Pass A transcripts per window, candidates, moments, the Gemini
3.5 Transcribe comparison responses, failed attempts and partials) and blind diarization
labels (`labeling/items.jsonl`). None of it can be regenerated identically.

What is NOT here: audio. Everything audio-shaped in work/ is re-cut from the recording.

This folder holds what was said in the lessons — keep it private. The repo is public and
only ever carries text-free summaries (tests/fixtures/).
"""


@dataclass
class Summary:
    copied: int = 0
    unchanged: int = 0
    not_archived: list[str] = field(default_factory=list)  # dates with no lesson folder


def main_work_dir() -> Path:
    """The main checkout's work/, even when called from a linked worktree.

    Worktrees share one .git but each has its own, usually empty, work/; the lesson data
    lives in the main checkout's, next to the shared .git directory.
    """
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent / "work"


def work_root(work_dir: Path) -> Path:
    """The work/ directory a --work-dir lives in, which is what the archive mirrors.

    `--work-dir work/hc9-2-verify` writes to work/hc9-2-verify/<date>/, and the archive
    keeps that as hc9-2-verify/<date>/ so it restores to the same place. A --work-dir
    outside any work/ is archived relative to itself.
    """
    work_dir = work_dir.resolve()
    for candidate in (work_dir, *work_dir.parents):
        if candidate.name == "work":
            return candidate
    return work_dir


def lesson_folder(recording: Path | None) -> Path | None:
    """The OneDrive lesson folder a recording sits in, if it sits in one."""
    if recording is None:
        return None
    recording = recording.resolve()
    return recording.parent if SOSO.resolve() in recording.parents else None


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


def lesson_date(relative: Path) -> str | None:
    """Which lesson a file under work/ belongs to: the first date folder in its path.

    That covers the pipeline's own work/<date>/ and ANY --work-dir under work/ — a
    listening kit cut with `--work-dir work/anything` lands in work/anything/<date>/labeling/
    and is found here without anyone having to list the folder name.
    """
    for part in relative.parts[:-1]:
        if LESSON_DATE.fullmatch(part):
            return part
    return UNDATED.get(relative.parts[0])


def plan(work: Path) -> dict[str, list[Path]]:
    """Every keepable file under `work`, grouped by lesson date."""
    by_date: dict[str, list[Path]] = {}
    for p in sorted(work.rglob("*")):
        if not p.is_file() or p.suffix.lower() in AUDIO or p.name == ".DS_Store":
            continue
        relative = p.relative_to(work)
        if "__pycache__" in relative.parts:
            continue
        date = lesson_date(relative)
        if date:
            by_date.setdefault(date, []).append(p)
    return by_date


def archive(work: Path, dates: set[str] | None = None, *, folder: Path | None = None,
            dry_run: bool = False, log=print) -> Summary:
    """Copy each lesson's keepable files into its OneDrive lesson folder.

    `folder` names the lesson folder directly (the pipeline knows where the recording is);
    otherwise it is looked up from the recordings. A lesson with no folder is reported and
    skipped — never guessed.
    """
    if folder is not None and (dates is None or len(dates) != 1):
        raise ValueError("an explicit lesson folder only makes sense for a single date")
    summary = Summary()
    work_plan = {d: files for d, files in plan(work).items() if dates is None or d in dates}
    lookup: dict[str, Path] | None = None  # scanning OneDrive is slow; only when needed
    for date, files in sorted(work_plan.items()):
        if folder is not None:
            target = folder
        else:
            lookup = recording_folders() if lookup is None else lookup
            target = lookup.get(date)
        if target is None:
            summary.not_archived.append(date)
            log(f"{date}: no lesson folder found under {SOSO} — NOT archived")
            continue
        dest_root = target / ARCHIVE_DIR / date
        new = 0
        for src in files:
            dest = dest_root / src.relative_to(work)
            if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
                summary.unchanged += 1
                continue
            new += 1
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        summary.copied += new
        where = dest_root.relative_to(SOSO.parent) if SOSO.parent in dest_root.parents else dest_root
        log(f"{date}: {new} new of {len(files)} files -> {where}")
        if not dry_run:
            (target / ARCHIVE_DIR / "README.md").write_text(README)
    return summary


def archive_lesson(date: str, work_dir: Path, recording: Path | None = None) -> None:
    """Archive one lesson after the pipeline or a listening session touched it.

    Warn-only, always: the archive is a safety copy, and failing to make one (OneDrive not
    mounted, a different machine) must never fail — or mask the error of — the run itself.
    """
    if not LESSON_DATE.fullmatch(date):
        return  # a synthetic smoke-test lesson; nothing worth keeping
    try:
        archive(work_root(work_dir), {date}, folder=lesson_folder(recording),
                log=lambda msg: print(f"[archive] {msg}", flush=True))
    except Exception as e:  # noqa: BLE001 — a safety copy must never take the run down
        print(f"[archive] warning: {date} not archived ({e.__class__.__name__}: {e}); "
              f"run `distill archive --date {date}` once OneDrive is available", flush=True)
