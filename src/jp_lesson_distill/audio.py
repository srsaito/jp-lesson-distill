"""ffmpeg helpers. Resolves a system ffmpeg if present, else the static binary
bundled in the imageio-ffmpeg wheel (so no brew install is required)."""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import cache
from pathlib import Path


@cache
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-y", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def prep_audio(src: Path, dst: Path) -> None:
    """Extract/downmix to mono 64 kbps AAC — small enough to upload, plenty for speech."""
    _run(["-i", str(src), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(dst)])


def clip_audio(src: Path, dst: Path, start: float, end: float) -> None:
    start = max(0.0, start)
    _run(["-ss", f"{start:.2f}", "-t", f"{max(1.0, end - start):.2f}", "-i", str(src), "-c", "copy", str(dst)])


def duration_seconds(path: Path) -> float:
    """Parse duration from `ffmpeg -i` output (no ffprobe in the bundled wheel)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", proc.stderr)
    if not m:
        raise RuntimeError(f"could not read duration of {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)
