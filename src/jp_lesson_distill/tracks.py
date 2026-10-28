"""Zoom per-participant tracks: find them, check them, and choose the Pass A path (ADR-0010).

Since 2026-10-08 Zoom writes one file per participant beside the mix:

    Soso_YYYYMMDD_class_audio.m4a           the mix (everything)
    Soso_YYYYMMDD_class_audio/
        audioSteveSaito1<meeting id>.m4a    Steven's mic
        audio末武創一2<meeting id>.m4a       Soso先生's mic
        audio末武創一3<meeting id>.m4a       his shared computer sound (textbook playback)

When both mic tracks are present and sound, Pass A transcribes per track. Anything else takes
the mixed path, which stays fully supported: lessons before 10/8, Zoom's setting off, or any
equipment problem (Steven's requirement, ADR-0010). The run log says which path and why.

Everything here is local (ffmpeg + numpy) and costs no API call.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio import duration_seconds, ffmpeg_exe

# Exact display-name prefixes, each with Zoom's per-person index. A looser match (audio*3*) once
# picked the wrong file, because the meeting id that follows is also full of digits.
ROLE_PREFIXES = {
    "student": "audioSteveSaito1",
    "teacher": "audio末武創一2",
    "share": "audio末武創一3",
}
MICS = ("student", "teacher")

LENGTH_TOLERANCE_S = 2.0   # every track is the mix's length; 10/8–10/25 matched to a frame
FRAME_S = 0.1              # energy resolution
SOUND_DB = -70.0           # any sound at all: Steven's soft はい／うーん sit at −33 to −55 dB
DIGITAL_SILENCE_DB = -100.0  # Zoom's noise suppression leaves exact zeros (−120 dB) between turns
MIC_SILENT_FRAC = 0.01     # a mic with sound in under 1% of the lesson recorded nobody
# With suppression on, 34–66% of a mic track is digital zero (10/8–10/25). Under this share the
# silence is room noise, i.e. suppression was off, and the −70 dB "any sound" line, the ghost
# filter and the snapping were never validated for that.
DIGITAL_SILENCE_MIN_FRAC = 0.05

# Pass A path the user can force; "auto" checks the tracks and falls back to the mix.
MODES = ("auto", "per-track", "mixed")


@dataclass
class TrackInfo:
    role: str
    path: str
    duration_s: float
    sound_frac: float
    digital_silence_frac: float


@dataclass
class Selection:
    path: str                    # "per-track" | "mixed"
    reason: str                  # one line for the run log
    tracks: dict[str, TrackInfo] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class TrackError(Exception):
    """A reason the tracks cannot be used. The message is what the log says."""


def tracks_dir(recording: Path) -> Path | None:
    """`…_class_audio/` beside `…_class_audio.m4a` (or beside the video, if that was given)."""
    for stem in (recording.stem, recording.stem.replace("_video", "_audio")):
        candidate = recording.with_name(stem)
        if candidate.is_dir():
            return candidate
    return None


def identify(folder: Path) -> dict[str, Path]:
    """Role → file, by exact prefix. A role with no file is absent; one with two is an error."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".m4a")
    found: dict[str, Path] = {}
    for role, prefix in ROLE_PREFIXES.items():
        hits = [p for p in files if p.name.startswith(prefix)]
        if len(hits) > 1:
            raise TrackError(f"{len(hits)} files match the {role} prefix {prefix!r}: "
                             f"{', '.join(p.name for p in hits)}")
        if hits:
            found[role] = hits[0]
    return found


def energy_db(path: Path) -> np.ndarray:
    """RMS level per 0.1 s frame, in dB (−120 for exact digital silence)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise TrackError(f"{path.name} could not be decoded: {proc.stderr.decode(errors='replace').strip()[:200]}")
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768
    per = int(16000 * FRAME_S)
    n = len(samples) // per
    if n == 0:
        raise TrackError(f"{path.name} has no audio")
    frames = samples[: n * per].reshape(n, per)
    return 10 * np.log10((frames ** 2).mean(1) + 1e-12)


def inspect(role: str, path: Path, mix_s: float) -> tuple[TrackInfo, np.ndarray]:
    """Check one track against the mix. Raises TrackError for anything that rules it out."""
    if path.stat().st_size == 0:
        raise TrackError(f"the {role} track {path.name} is empty (0 bytes)")
    try:
        length = duration_seconds(path)
    except RuntimeError as exc:
        raise TrackError(f"the {role} track {path.name} is unreadable ({exc})") from exc
    if abs(length - mix_s) > LENGTH_TOLERANCE_S:
        raise TrackError(f"the {role} track {path.name} is {length:.1f}s long and the mix is "
                         f"{mix_s:.1f}s; tracks recorded with the mix match it to within "
                         f"{LENGTH_TOLERANCE_S:g}s")
    db = energy_db(path)
    info = TrackInfo(role=role, path=str(path), duration_s=round(length, 1),
                     sound_frac=round(float((db > SOUND_DB).mean()), 4),
                     digital_silence_frac=round(float((db <= DIGITAL_SILENCE_DB).mean()), 4))
    return info, db


def select(recording: Path, mix_s: float, mode: str = "auto", work: Path | None = None) -> Selection:
    """Choose per-track or mixed Pass A for this lesson, and say why.

    `mode="mixed"` skips the checks. `mode="per-track"` runs them and raises SystemExit
    instead of falling back, because someone who forced the tracks wants to hear that they
    could not be used, not a quiet mixed run. With `work`, the track energy is cached as
    `tracks/energy.npz` for the per-track stages and the decision is saved as
    `tracks/selection.json`.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "mixed":
        sel = Selection("mixed", "the mixed path was asked for (--mixed)")
    else:
        try:
            sel = _check(recording, mix_s, work)
        except TrackError as exc:
            if mode == "per-track":
                raise SystemExit(f"[tracks] --per-track was asked for, but the tracks cannot be used: {exc}")
            sel = Selection("mixed", f"falling back to the mixed recording: {exc}")
    if work is not None:
        out = work / "tracks"
        out.mkdir(parents=True, exist_ok=True)
        (out / "selection.json").write_text(sel.to_json())
    return sel


def _check(recording: Path, mix_s: float, work: Path | None) -> Selection:
    folder = tracks_dir(recording)
    if folder is None:
        raise TrackError(f"no per-participant tracks beside {recording.name} (lessons before "
                         "2026-10-08 have none, and Zoom only writes them with "
                         "「参加者のオーディオファイルを個別にレコーディングする」 on)")
    found = identify(folder)
    missing = [r for r in MICS if r not in found]
    if missing:
        names = ", ".join(p.name for p in sorted(folder.glob("*.m4a"))) or "no .m4a files"
        raise TrackError(f"the {' and '.join(missing)} mic track is missing from {folder.name}/ "
                         f"(found: {names}); a partial set takes the mixed path rather than "
                         "guessing who the missing voice was")

    tracks: dict[str, TrackInfo] = {}
    energies: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for role in (*MICS, "share"):
        if role not in found:
            continue
        info, db = inspect(role, found[role], mix_s)
        tracks[role], energies[role] = info, db
    for role in MICS:
        if tracks[role].sound_frac < MIC_SILENT_FRAC:
            raise TrackError(f"the {role} mic track has sound in only {tracks[role].sound_frac:.1%} "
                             "of the lesson; a mic that recorded nobody cannot carry their lines")
        if tracks[role].digital_silence_frac < DIGITAL_SILENCE_MIN_FRAC:
            warnings.append(
                f"the {role} track's silence is not digital zero ({tracks[role].digital_silence_frac:.0%} "
                "of frames, against 34–66% with Zoom's noise suppression on). Suppression may have "
                "been off; the per-track thresholds were only measured with it on")
    if "share" not in found:
        warnings.append("no shared-sound track (audio末武創一3…): textbook playback, if any, was not "
                        "recorded separately")
    elif tracks["share"].sound_frac < MIC_SILENT_FRAC:
        warnings.append(f"the shared-sound track is silent ({tracks['share'].sound_frac:.1%} of the "
                        "lesson has sound): no playback detected, which is normal for a lesson "
                        "without textbook audio")
    if work is not None:
        (work / "tracks").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(work / "tracks" / "energy.npz", **energies)
    return Selection("per-track", f"per-participant tracks found and usable in {folder.name}/",
                     tracks=tracks, warnings=warnings)
