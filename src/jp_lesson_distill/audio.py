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


def window_starts(total: float, win_s: float, overlap_s: float) -> list[float]:
    """Start offsets of overlapping windows covering `total` seconds.

    Pure planning half of `split_windows` (no ffmpeg), so it can be unit-tested.
    An empty list means "no windowing" — the caller uses the file as-is.
    """
    if win_s <= 0 or total <= win_s:
        return []
    overlap_s = max(0.0, min(overlap_s, win_s / 2))
    stride = win_s - overlap_s
    starts, s = [], 0.0
    while s < total:
        starts.append(s)
        s += stride
    # A trailing sliver shorter than the overlap carries no speech the previous
    # window doesn't already have — drop it and let that window run to the end.
    if len(starts) > 1 and total - starts[-1] <= overlap_s:
        starts.pop()
    return starts


def split_windows(src: Path, out_dir: Path, win_s: float = 20 * 60,
                  overlap_s: float = 30.0, total: float | None = None) -> list[tuple[Path, float]]:
    """Cut `src` into overlapping windows by stream copy. Returns [(path, offset_s)].

    Recordings at or under one window (and win_s <= 0) come back as [(src, 0.0)] —
    no copy, no windowing. Existing window files are reused.
    """
    total = duration_seconds(src) if total is None else total
    starts = window_starts(total, win_s, overlap_s)
    if not starts:
        return [(src, 0.0)]
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    for i, start in enumerate(starts, start=1):
        dst = out_dir / f"w{i:02d}{src.suffix}"
        if not dst.exists():
            clip_audio(src, dst, start, min(total, start + win_s))
        windows.append((dst, start))
    return windows
