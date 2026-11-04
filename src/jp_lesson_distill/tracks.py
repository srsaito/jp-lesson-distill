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

Which file is whose (jld-8nv.7). The digit after the display name is NOT fixed: 10/8–10/26 were
SteveSaito1 / 末武創一2 / 末武創一3, and 11/3 was SteveSaito2 / 末武創一1 / 末武創一3 (it looks
like join order). So roles are worked out, not looked up:

1. Group the files by person: the display name with the trailing digits (Zoom's index plus the
   meeting id) removed.
2. A person with two files is the one sharing computer sound: the quieter file is the share
   track, the other their mic. Measured: share 1.5–5% of the lesson has sound, mics 25–35%.
3. Student vs teacher by display-name fragments (STUDENT_NAMES / TEACHER_NAMES); if only one
   side matches, the other person takes the other role; failing that, the sharer is the
   teacher.
4. Anything still ambiguous (new display names, a guest, an odd file count) goes to an optional
   `classify` callback, which in production is a Gemini call that listens to a short sample of
   every track (`classify_with_gemini`). Its answer must be confident and consistent, or the
   lesson takes the mixed path as before.

Steps 1–3 are local (ffmpeg + numpy) and cost no API call; step 4 runs only when they cannot
decide, and its answer is cached in the work folder.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio import duration_seconds, ffmpeg_exe

# Display-name fragments, matched case-insensitively against the name with spaces removed.
# Matching is a convenience, not a requirement: an unmatched name goes to the energy rule and
# then to the classifier (step 4 in the module docstring).
STUDENT_NAMES = ("stevesaito", "steven", "steve", "saito", "スティーブ", "齋藤", "斎藤")
TEACHER_NAMES = ("末武", "創一", "soso", "そそ", "ソソ", "suetake", "souichi", "soichi")
MICS = ("student", "teacher")
ROLES = (*MICS, "share")
# A share track has sound in 1.5–5% of a lesson (10/8–11/3); a mic in 25–35%. A person's quieter
# file is their share track only if it is clearly quieter than their other file.
SHARE_MAX_SOUND = 0.15
SHARE_RATIO = 3.0
CLASSIFY_MIN_CONFIDENCE = 0.8

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
    assigned_by: str = ""        # how the roles were worked out: "names", "energy", "classifier"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class TrackError(Exception):
    """A reason the tracks cannot be used. The message is what the log says."""


class MissingMic(TrackError):
    """A mic is plainly absent: no classifier can supply a voice that was not recorded."""


# classify(files, work) -> {"student": Path, "teacher": Path, "share": Path | None,
#                            "confidence": float, "reason": str}
Classifier = "Callable[[list[Path], Path | None], dict]"


def tracks_dir(recording: Path) -> Path | None:
    """`…_class_audio/` beside `…_class_audio.m4a` (or beside the video, if that was given)."""
    for stem in (recording.stem, recording.stem.replace("_video", "_audio")):
        candidate = recording.with_name(stem)
        if candidate.is_dir():
            return candidate
    return None


@dataclass
class TrackFile:
    path: Path
    person: str          # display name, trailing digits removed
    index: str           # Zoom's per-person index, when the meeting id could be separated
    sound_frac: float = 0.0


def _person(path: Path) -> tuple[str, str]:
    """('SteveSaito', '21936486817') from audioSteveSaito21936486817.m4a."""
    m = re.match(r"^(?:audio)?(.*?)(\d*)$", path.stem)
    return m.group(1), m.group(2)


def parse_files(folder: Path) -> list[TrackFile]:
    """Every .m4a in the folder, grouped by person, with Zoom's index split off when possible.

    The meeting id is the digits every file ends with; what is left of each trailing digit
    run is the index. A file from another meeting breaks the common suffix, which only costs
    the index (it is informational), never the grouping.
    """
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".m4a")
    parsed = [(p, *_person(p)) for p in files]
    runs = [r for _, _, r in parsed if r]
    common = ""
    if len(runs) > 1:
        k = 0
        while k < min(map(len, runs)) and len({r[-1 - k] for r in runs}) == 1:
            k += 1
        common = runs[0][len(runs[0]) - k:] if k else ""
    out = []
    for p, name, run in parsed:
        index = run[: len(run) - len(common)] if common and run.endswith(common) else ""
        out.append(TrackFile(path=p, person=name, index=index))
    return out


def _matches(name: str, fragments: tuple[str, ...]) -> bool:
    key = name.replace(" ", "").casefold()
    return any(f.casefold() in key for f in fragments)


def identify(folder: Path, sound: dict[Path, float] | None = None) -> tuple[dict[str, Path], str]:
    """Role -> file, and how it was decided ("names" or "energy").

    `sound` is each file's fraction of frames with sound (from `inspect`); without it, a
    person with two files cannot be split into mic and share, which is then ambiguous. Raises
    TrackError when the files do not settle it; `_check` then asks the classifier, if any.
    """
    files = parse_files(folder)
    if sound:
        for f in files:
            f.sound_frac = sound.get(f.path, 0.0)
    people: dict[str, list[TrackFile]] = {}
    for f in files:
        people.setdefault(f.person, []).append(f)
    listing = ", ".join(f.path.name for f in files) or "no .m4a files"

    mics: dict[str, TrackFile] = {}
    share: TrackFile | None = None
    for person, own in people.items():
        if len(own) == 1:
            mics[person] = own[0]
        elif len(own) == 2 and sound:
            quiet, loud = sorted(own, key=lambda f: f.sound_frac)
            # The mic must sound like a mic (more than any share track ever has); otherwise a
            # silent mic beside a share track would pass the share track off as the mic.
            if (quiet.sound_frac <= SHARE_MAX_SOUND < loud.sound_frac
                    and loud.sound_frac >= SHARE_RATIO * quiet.sound_frac):
                if share is not None:
                    raise TrackError(f"two people seem to have shared sound ({listing})")
                mics[person], share = loud, quiet
            elif loud.sound_frac <= SHARE_MAX_SOUND:
                raise TrackError(f"{person} has two files and neither sounds like a mic (sound in "
                                 f"{quiet.sound_frac:.1%} and {loud.sound_frac:.1%} of the lesson; "
                                 f"{listing}); a mic that recorded nobody cannot carry their lines")
            else:
                raise TrackError(f"{person} has two files that both sound like a mic "
                                 f"({listing}); cannot tell which is theirs")
        else:
            raise TrackError(f"{person} has {len(own)} files ({listing}); cannot tell which is "
                             "their mic")

    # Zoom numbers a meeting's files 1..N (the order is join order, not role). A gap means a
    # file is missing, e.g. Soso先生's shared sound with his own mic absent; without this, that
    # lone share track would pass as his mic.
    indexes = sorted(int(f.index) for f in files if f.index.isdigit())
    if len(indexes) == len(files) and indexes and indexes != list(range(1, len(files) + 1)):
        gap = sorted(set(range(1, max(indexes) + 1)) - set(indexes))
        raise MissingMic(f"a mic track is missing from {folder.name}/: Zoom numbers the files 1 to "
                         f"N and {', '.join(map(str, gap)) or 'a number'} is absent (found: {listing}); "
                         "a partial set takes the mixed path rather than guessing who the missing "
                         "voice was")
    if len(mics) > 2:
        raise TrackError(f"{len(mics)} people have tracks ({listing}); expected Steven and "
                         "Soso先生 only")
    student = [p for p in mics if _matches(p, STUDENT_NAMES)]
    teacher = [p for p in mics if _matches(p, TEACHER_NAMES) and p not in student]
    how = "names"
    if len(mics) == 2:
        a, b = mics
        if len(student) == 1 and not teacher:
            teacher = [b if student[0] == a else a]
        elif len(teacher) == 1 and not student:
            student = [b if teacher[0] == a else a]
        elif not student and not teacher and share is not None:
            sharer = next(p for p in people if share in people[p])
            teacher, student = [sharer], [b if sharer == a else a]
            how = "energy"
    if len(student) > 1 or len(teacher) > 1:
        raise TrackError(f"the display names match more than one person ({listing})")
    if not student and not teacher:
        if len(mics) == 2:
            raise TrackError(f"cannot tell who is who from the display names ({listing})")
        # One person only: the other mic is missing (handled below with the usual message).
    missing = [r for r, who in (("student", student), ("teacher", teacher)) if not who]
    if missing:
        raise MissingMic(f"the {' and '.join(missing)} mic track is missing from {folder.name}/ "
                         f"(found: {listing}); a partial set takes the mixed path rather than "
                         "guessing who the missing voice was")
    found = {"student": mics[student[0]].path, "teacher": mics[teacher[0]].path}
    if share is not None:
        found["share"] = share.path
    return found, how


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


def select(recording: Path, mix_s: float, mode: str = "auto", work: Path | None = None,
           classify=None) -> Selection:
    """Choose per-track or mixed Pass A for this lesson, and say why.

    `mode="mixed"` skips the checks. `mode="per-track"` runs them and raises SystemExit
    instead of falling back, because someone who forced the tracks wants to hear that they
    could not be used, not a quiet mixed run. With `work`, the track energy is cached as
    `tracks/energy.npz` for the per-track stages and the decision is saved as
    `tracks/selection.json`. `classify` is consulted only when the files cannot settle who is
    who (see the module docstring); without it, that case takes the mixed path.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "mixed":
        sel = Selection("mixed", "the mixed path was asked for (--mixed)")
    else:
        try:
            sel = _check(recording, mix_s, work, classify)
        except TrackError as exc:
            if mode == "per-track":
                raise SystemExit(f"[tracks] --per-track was asked for, but the tracks cannot be used: {exc}")
            sel = Selection("mixed", f"falling back to the mixed recording: {exc}")
    if work is not None:
        out = work / "tracks"
        out.mkdir(parents=True, exist_ok=True)
        (out / "selection.json").write_text(sel.to_json())
    return sel


def _check(recording: Path, mix_s: float, work: Path | None, classify=None) -> Selection:
    folder = tracks_dir(recording)
    if folder is None:
        raise TrackError(f"no per-participant tracks beside {recording.name} (lessons before "
                         "2026-10-08 have none, and Zoom only writes them with "
                         "「参加者のオーディオファイルを個別にレコーディングする」 on)")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".m4a")
    if not files:
        raise MissingMic(f"the student and teacher mic track is missing from {folder.name}/ "
                         "(found: no .m4a files)")

    # Every file is checked before roles are assigned: the energy is what tells a share track
    # from a mic, and a bad file is a reason to distrust the set.
    infos: dict[Path, TrackInfo] = {}
    dbs: dict[Path, np.ndarray] = {}
    for path in files:
        infos[path], dbs[path] = inspect(_person(path)[0] or path.stem, path, mix_s)

    warnings: list[str] = []
    try:
        found, how = identify(folder, {p: i.sound_frac for p, i in infos.items()})
    except MissingMic:
        raise
    except TrackError as exc:
        if classify is None:
            raise
        print(f"[tracks] {exc}; asking the classifier")
        try:
            answer = classify(files, work)
        except Exception as cexc:  # noqa: BLE001 — the mixed path is the required fallback
            raise TrackError(f"{exc}, and the classifier failed ({type(cexc).__name__}: "
                             f"{str(cexc)[:200]})") from cexc
        found = _accept(answer, files)
        how = "classifier"
        warnings.append(f"roles assigned by the classifier (confidence "
                        f"{answer.get('confidence', 0):.2f}): {answer.get('reason', '')}")

    tracks: dict[str, TrackInfo] = {}
    energies: dict[str, np.ndarray] = {}
    for role, path in found.items():
        tracks[role] = TrackInfo(**{**asdict(infos[path]), "role": role})
        energies[role] = dbs[path]
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
        warnings.append("no shared-sound track: textbook playback, if any, was not "
                        "recorded separately")
    elif tracks["share"].sound_frac < MIC_SILENT_FRAC:
        warnings.append(f"the shared-sound track is silent ({tracks['share'].sound_frac:.1%} of the "
                        "lesson has sound): no playback detected, which is normal for a lesson "
                        "without textbook audio")
    if work is not None:
        (work / "tracks").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(work / "tracks" / "energy.npz", **energies)
    names = ", ".join(f"{r}={Path(t.path).name}" for r, t in tracks.items())
    return Selection("per-track", f"per-participant tracks found and usable in {folder.name}/ "
                     f"(roles by {how}: {names})", tracks=tracks, warnings=warnings,
                     assigned_by=how)


def _accept(answer: dict, files: list[Path]) -> dict[str, Path]:
    """The classifier's answer, if it is confident and consistent; TrackError otherwise."""
    by_name = {p.name: p for p in files}
    conf = float(answer.get("confidence") or 0)
    if conf < CLASSIFY_MIN_CONFIDENCE:
        raise TrackError(f"the classifier was not sure who is who (confidence {conf:.2f}: "
                         f"{answer.get('reason', '')})")
    found: dict[str, Path] = {}
    for role in ROLES:
        name = answer.get(role)
        if not name:
            continue
        name = Path(str(name)).name
        if name not in by_name:
            raise TrackError(f"the classifier named a file that is not there ({name!r})")
        found[role] = by_name[name]
    missing = [r for r in MICS if r not in found]
    if missing:
        raise TrackError(f"the classifier found no {' or '.join(missing)} mic "
                         f"({answer.get('reason', '')})")
    if len(set(found.values())) != len(found):
        raise TrackError("the classifier gave one file two roles")
    return found


# --- step 4: ask Gemini who is who (only when the files cannot settle it) ---

CLASSIFY_CLIP_S = 40.0


def _loudest_window(db: np.ndarray, span_s: float) -> float:
    """Start (s) of the `span_s` stretch with the most frames above the sound line."""
    w = int(span_s / FRAME_S)
    act = (db > SOUND_DB).astype(np.int32)
    if len(act) <= w:
        return 0.0
    sums = np.convolve(act, np.ones(w, dtype=np.int32), mode="valid")
    return float(int(sums.argmax()) * FRAME_S)


def classify_with_gemini(client_factory, model: str):
    """A `classify` callback for `select`: listen to a sample of every track, assign roles.

    `client_factory` builds the Gemini client lazily, so a lesson whose files settle the
    roles makes no call. The answer is cached as `tracks/roles_classifier.json`, keyed by the
    file names, so a rerun of the same lesson asks once.
    """
    def classify(files: list[Path], work: Path | None) -> dict:
        from pydantic import BaseModel, Field

        from . import gemini
        from .audio import clip_audio

        key = sorted(p.name for p in files)
        cache = work / "tracks" / "roles_classifier.json" if work is not None else None
        if cache is not None and cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("files") == key:
                print(f"[tracks] classifier answer cached: {cache}")
                return saved["answer"]

        class TrackRoles(BaseModel):
            student: str = Field(description="file name of the learner's microphone track")
            teacher: str = Field(description="file name of the teacher's microphone track")
            share: str | None = Field(None, description="file name of shared computer sound "
                                      "(recorded textbook audio), or null if none")
            confidence: float = Field(description="0 to 1")
            reason: str = Field(description="one sentence: what you heard that decided it")

        scratch = (work / "tracks" / "classify") if work is not None else Path(
            __import__("tempfile").mkdtemp())
        scratch.mkdir(parents=True, exist_ok=True)
        contents: list = []
        for k, path in enumerate(files):
            db = energy_db(path)
            start = _loudest_window(db, CLASSIFY_CLIP_S)
            clip = scratch / f"track{k}.m4a"
            clip_audio(path, clip, start, start + CLASSIFY_CLIP_S)
            contents += [f"File {path.name} (sound in {(db > SOUND_DB).mean():.0%} of the "
                         f"lesson; sample from {start:.0f}s):", gemini.audio_part(clip)]
        contents.append(
            "These are Zoom's per-participant recordings of one online Japanese lesson: one file "
            "per participant microphone, plus possibly one file of computer sound a participant "
            "shared (recorded textbook dialogues read by voice actors, usually present only a few "
            "percent of the lesson). The participants are a native-speaker Japanese teacher and "
            "an adult English-speaking learner of Japanese (Steven), who hesitates, makes "
            "mistakes and sometimes switches to English. File names carry Zoom display names and "
            "numbers that are NOT reliable. Assign each role to exactly one file name, copied "
            "exactly. If you cannot tell, give a low confidence.")
        answer = gemini.generate(client_factory(), model, contents, TrackRoles).model_dump()
        print(f"[tracks] classifier: student={answer['student']} teacher={answer['teacher']} "
              f"share={answer['share']} (confidence {answer['confidence']:.2f}): {answer['reason']}")
        if cache is not None:
            cache.write_text(json.dumps({"files": key, "answer": answer}, ensure_ascii=False, indent=2))
        return answer

    return classify


# --- relabelling the mixed transcript from the tracks (jld-ec6) ---
#
# The rule that repaired 10/14 (12 -> 22 moments), kept exactly as measured: a segment runs from
# its start to the next segment's start; it is the student's when Steven's mic is active over
# 30% of that span and twice as active as the teacher's, the teacher's likewise, and playback
# when the share track is active over half of it. Anything else is left alone: two voices in
# one segment (jld-beb) or a boundary in the wrong place, which no relabel can fix.
RELABEL_ACTIVE_DB = -45.0
RELABEL_MIN_ACTIVE = 0.3
RELABEL_DOMINANCE = 2.0
PLAYBACK_SHARE = 0.5


def speaker_from_energy(energy: dict[str, np.ndarray], a: float, b: float) -> str | None:
    """'student', 'teacher', 'played', or None when the tracks cannot say."""
    fa, fb = int(a / FRAME_S), int(b / FRAME_S)
    if fb <= fa:
        return None

    def active(role: str) -> float:
        db = energy.get(role)
        if db is None or fa >= len(db):
            return 0.0
        return float((db[fa:fb] > RELABEL_ACTIVE_DB).mean())

    st, te, sh = active("student"), active("teacher"), active("share")
    if sh > PLAYBACK_SHARE:
        return "played"
    if st > RELABEL_MIN_ACTIVE and st > RELABEL_DOMINANCE * te:
        return "student"
    if te > RELABEL_MIN_ACTIVE and te > RELABEL_DOMINANCE * st:
        return "teacher"
    return None


@dataclass
class WindowCheck:
    window: int
    start_s: float
    end_s: float
    agree: int = 0      # labels the tracks confirm
    flipped: int = 0    # labels the tracks contradict (and that were flipped)
    undecided: int = 0  # segments the tracks cannot settle
    played: int = 0     # segments over textbook playback (left as they are; jld-lg6)

    @property
    def agreement(self) -> float | None:
        n = self.agree + self.flipped
        return self.agree / n if n else None


def relabel(segments: list, energy: dict[str, np.ndarray], total: float,
            bounds: list[tuple[float, float]]) -> tuple[list, list[WindowCheck]]:
    """Flip every label the tracks clearly contradict; report per window what they said.

    `segments` are production Segments with absolute "MM:SS" starts; `bounds` are the
    stretches each Pass A window owns after the merge, because a label inversion is a
    per-window property (each window is diarized by its own call).
    """
    from .models import parse_ts

    starts = [parse_ts(s.start) for s in segments]
    checks = [WindowCheck(i + 1, round(a, 1), round(b, 1)) for i, (a, b) in enumerate(bounds)]
    out = []
    for i, seg in enumerate(segments):
        a = starts[i]
        b = starts[i + 1] if i + 1 < len(segments) else total
        check = next((c for c in checks if c.start_s <= a < c.end_s), checks[-1])
        truth = speaker_from_energy(energy, a, b)
        if truth is None:
            check.undecided += 1
        elif truth == "played":
            check.played += 1
        elif truth == seg.speaker:
            check.agree += 1
        else:
            check.flipped += 1
            seg = seg.model_copy(update={"speaker": truth})
        out.append(seg)
    return out, checks


SNAP_S = 3.0  # how far a model's start may sit from its own track's sound onset


def snap_start(db: np.ndarray, s: float) -> float:
    """Move a model-given start onto its own track's sound onset (0.1 s resolution).

    Gemini's starts are whole seconds and model-generated; the track says when the voice
    actually began. Inside a stretch of sound, the start of that stretch wins if it is within
    SNAP_S; otherwise the nearest onset within SNAP_S; otherwise the model's start stands.
    """
    act = db > SOUND_DB
    f = int(s / FRAME_S)
    if 0 <= f < len(act) and act[f]:
        g = f
        while g > 0 and act[g - 1]:
            g -= 1
        if s - g * FRAME_S <= SNAP_S:
            return round(g * FRAME_S, 1)
    lo = max(1, int((s - SNAP_S) / FRAME_S))
    hi = min(len(act), int((s + SNAP_S) / FRAME_S) + 1)
    onsets = [k for k in range(lo, hi) if act[k] and not act[k - 1]]
    if not onsets:
        return s
    return round(min(onsets, key=lambda k: abs(k * FRAME_S - s)) * FRAME_S, 1)


def selected_paths(work: Path) -> dict[str, Path]:
    """Role -> file, as `select` decided and saved it (so later stages never re-guess)."""
    saved = json.loads((work / "tracks" / "selection.json").read_text())
    return {role: Path(t["path"]) for role, t in saved.get("tracks", {}).items()}


def load_energy(work: Path) -> dict[str, np.ndarray] | None:
    """The track energy `select` saved, or None when the tracks were not usable."""
    path = work / "tracks" / "energy.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        energy = {k: data[k] for k in data.files}
    return energy if all(m in energy for m in MICS) else None
