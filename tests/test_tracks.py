"""Choosing the Pass A path from Zoom's per-participant tracks (ADR-0010, jld-8nv.1).

The mixed recording is the required fallback (Steven, 2026-10-26): every way the tracks can
be unusable must end in the mixed path with a reason, never in a failed run. One test per
trigger in ADR-0010's list. The tracks are a few seconds of synthetic audio made with the
bundled ffmpeg: a tone where someone speaks and exact digital zeros elsewhere, which is what
Zoom's noise suppression writes between turns.
"""
import json
import subprocess

import pytest

from jp_lesson_distill.audio import ffmpeg_exe
from jp_lesson_distill.tracks import identify, select, tracks_dir

LEN = 6.0
STEVEN = "audioSteveSaito11306470538.m4a"
TEACHER = "audio末武創一21306470538.m4a"
SHARE = "audio末武創一31306470538.m4a"


def make(path, speech=((1.0, 3.0),), length=LEN, noise=0.0):
    """An m4a with a tone during `speech` spans, digital silence (or faint noise) elsewhere."""
    on = "+".join(f"between(t,{a},{b})" for a, b in speech) or "0"
    expr = f"if({on},0.3*sin(2*PI*440*t),0)"
    if noise:
        expr += f"+{noise}*(random(0)-0.5)"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"aevalsrc='{expr}':s=16000:d={length}", "-c:a", "aac", "-b:a", "64k",
                    str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """Each kind of track encoded once; tests copy what they need."""
    d = tmp_path_factory.mktemp("clips")
    return {
        "mix": make(d / "mix.m4a", speech=((0.5, 5.5),)),
        "steven": make(d / "steven.m4a", speech=((1.0, 2.5),)),
        "teacher": make(d / "teacher.m4a", speech=((3.0, 5.0),)),
        "share": make(d / "share.m4a", speech=((5.2, 5.8),)),
        "silent": make(d / "silent.m4a", speech=()),
        "short": make(d / "short.m4a", length=LEN - 3),
        "noisy": make(d / "noisy.m4a", speech=((1.0, 2.5),), noise=0.002),
    }


def lesson(tmp_path, clips, **files):
    """Soso_20991025_class_audio.m4a plus a track folder holding `files` (name -> clip key or bytes)."""
    rec = tmp_path / "Soso_20991025_class_audio.m4a"
    rec.write_bytes(clips["mix"].read_bytes())
    if files:
        folder = tmp_path / "Soso_20991025_class_audio"
        folder.mkdir()
        for name, src in files.items():
            (folder / name).write_bytes(src if isinstance(src, bytes) else clips[src].read_bytes())
    return rec


FULL = {STEVEN: "steven", TEACHER: "teacher", SHARE: "share"}


def test_a_full_set_selects_per_track_and_saves_the_decision_and_the_energy(tmp_path, clips):
    rec = lesson(tmp_path, clips, **FULL)
    work = tmp_path / "work"
    sel = select(rec, LEN, work=work)
    assert sel.path == "per-track", sel.reason
    assert set(sel.tracks) == {"student", "teacher", "share"}
    assert sel.warnings == []
    assert sel.tracks["student"].digital_silence_frac > 0.5  # the tone is 1.5 s of 6
    saved = json.loads((work / "tracks" / "selection.json").read_text())
    assert saved["path"] == "per-track"
    assert (work / "tracks" / "energy.npz").exists()


def test_no_track_folder_is_the_mixed_path_for_lessons_before_october(tmp_path, clips):
    sel = select(lesson(tmp_path, clips), LEN)
    assert sel.path == "mixed"
    assert "no per-participant tracks" in sel.reason


@pytest.mark.parametrize("drop,who", [(TEACHER, "teacher"), (STEVEN, "student")])
def test_a_missing_mic_takes_the_mixed_path_for_the_whole_lesson(tmp_path, clips, drop, who):
    files = {k: v for k, v in FULL.items() if k != drop}
    sel = select(lesson(tmp_path, clips, **files), LEN)
    assert sel.path == "mixed"
    assert "mic track is missing" in sel.reason


def test_a_track_of_the_wrong_length_is_not_from_this_recording(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **{**FULL, TEACHER: "short"}), LEN)
    assert sel.path == "mixed"
    assert "long and the mix is" in sel.reason


def test_an_empty_track_falls_back(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **{**FULL, STEVEN: b""}), LEN)
    assert sel.path == "mixed"
    assert "empty" in sel.reason


def test_a_corrupt_track_falls_back(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **{**FULL, STEVEN: b"not an m4a at all" * 100}), LEN)
    assert sel.path == "mixed"
    assert "unreadable" in sel.reason or "could not be decoded" in sel.reason


def test_a_mic_that_recorded_nobody_falls_back(tmp_path, clips):
    # Beside his share track, a silent teacher mic must not let the share track pass as the mic.
    sel = select(lesson(tmp_path, clips, **{**FULL, TEACHER: "silent"}), LEN)
    assert sel.path == "mixed"
    assert "neither sounds like a mic" in sel.reason

    other = tmp_path / "b"
    other.mkdir()
    sel = select(lesson(other, clips, **{STEVEN: "steven", TEACHER: "silent"}), LEN)
    assert sel.path == "mixed"
    assert "teacher mic track has sound in only" in sel.reason


def test_two_files_for_one_role_is_ambiguous_and_falls_back(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **FULL, **{"audioSteveSaito19999999999.m4a": "steven"}), LEN)
    assert sel.path == "mixed"
    assert "two files that both sound like a mic" in sel.reason


def test_a_missing_or_silent_share_track_only_warns(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **{STEVEN: "steven", TEACHER: "teacher"}), LEN)
    assert sel.path == "per-track"
    assert any("no shared-sound track" in w for w in sel.warnings)

    other = tmp_path / "b"
    other.mkdir()
    sel = select(lesson(other, clips, **{**FULL, SHARE: "silent"}), LEN)
    assert sel.path == "per-track"
    assert any("no playback detected" in w for w in sel.warnings)


def test_silence_that_is_not_digital_zero_warns(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **{**FULL, STEVEN: "noisy"}), LEN)
    assert sel.path == "per-track"
    assert any("not digital zero" in w for w in sel.warnings)


def test_digits_in_the_meeting_id_never_decide_a_role(tmp_path):
    # A loose glob for the share track (audio*3*) once matched the teacher's file, whose
    # meeting id is full of 3s. Roles now come from the display name, never the digits.
    folder = tmp_path / "x"
    folder.mkdir()
    for name in ("audio末武創一23333333333.m4a", "audioSteveSaito13333333333.m4a"):
        (folder / name).write_bytes(b"")
    found, how = identify(folder)
    assert found["teacher"].name.startswith("audio末武創一2")
    assert "share" not in found and how == "names"


# --- jld-8nv.7: Zoom renumbers the files; roles are worked out, not looked up ---

RENUMBERED = {  # 2026-11-03: Steven joined second, so every index moved
    "audioSteveSaito21936486817.m4a": "steven",
    "audio末武創一11936486817.m4a": "teacher",
    "audio末武創一31936486817.m4a": "share",
}


def test_the_11_3_renumbering_still_selects_per_track(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **RENUMBERED), LEN)
    assert sel.path == "per-track", sel.reason
    assert sel.assigned_by == "names"
    assert sel.tracks["student"].path.endswith("audioSteveSaito21936486817.m4a")
    assert sel.tracks["teacher"].path.endswith("audio末武創一11936486817.m4a")
    assert sel.tracks["share"].path.endswith("audio末武創一31936486817.m4a")


def test_the_share_track_is_the_sharers_quieter_file_whatever_its_number(tmp_path, clips):
    # Swap which of his two files carries which number: the quiet one is still the share.
    files = {"audioSteveSaito11.m4a": "steven", "audio末武創一31.m4a": "teacher",
             "audio末武創一21.m4a": "share"}
    sel = select(lesson(tmp_path, clips, **files), LEN)
    assert sel.path == "per-track", sel.reason
    assert sel.tracks["teacher"].path.endswith("audio末武創一31.m4a")
    assert sel.tracks["share"].path.endswith("audio末武創一21.m4a")


def test_one_known_name_is_enough(tmp_path, clips):
    files = {"audioSteven Saito21234567890.m4a": "steven", "audioiPad11234567890.m4a": "teacher"}
    sel = select(lesson(tmp_path, clips, **files), LEN)
    assert sel.path == "per-track", sel.reason
    assert sel.tracks["teacher"].path.endswith("audioiPad11234567890.m4a")


def test_unknown_names_fall_back_to_who_is_sharing(tmp_path, clips):
    files = {"audioGuest11234567890.m4a": "steven", "audioZoomUser21234567890.m4a": "teacher",
             "audioZoomUser31234567890.m4a": "share"}
    sel = select(lesson(tmp_path, clips, **files), LEN)
    assert sel.path == "per-track", sel.reason
    assert sel.assigned_by == "energy"
    assert sel.tracks["teacher"].path.endswith("audioZoomUser21234567890.m4a")
    assert sel.tracks["student"].path.endswith("audioGuest11234567890.m4a")


UNKNOWN = {"audioGuest11234567890.m4a": "steven", "audioiPhone21234567890.m4a": "teacher"}


def test_unknown_names_and_no_share_without_a_classifier_fall_back(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **UNKNOWN), LEN)
    assert sel.path == "mixed"
    assert "cannot tell who is who" in sel.reason


def test_the_classifier_settles_what_the_files_cannot(tmp_path, clips):
    asked = []

    def classify(files, work):
        asked.append(sorted(p.name for p in files))
        return {"student": "audioGuest11234567890.m4a", "teacher": "audioiPhone21234567890.m4a",
                "share": None, "confidence": 0.95, "reason": "learner hesitates in English"}

    sel = select(lesson(tmp_path, clips, **UNKNOWN), LEN, classify=classify)
    assert sel.path == "per-track", sel.reason
    assert sel.assigned_by == "classifier"
    assert sel.tracks["student"].path.endswith("audioGuest11234567890.m4a")
    assert any("classifier" in w for w in sel.warnings)
    assert len(asked) == 1


def test_the_classifier_is_not_asked_when_the_names_settle_it(tmp_path, clips):
    def classify(files, work):
        raise AssertionError("should not be called")

    assert select(lesson(tmp_path, clips, **RENUMBERED), LEN, classify=classify).path == "per-track"


@pytest.mark.parametrize("answer,why", [
    ({"student": "audioGuest11234567890.m4a", "teacher": "audioiPhone21234567890.m4a",
      "confidence": 0.4, "reason": "both sound fluent"}, "not sure"),
    ({"student": "audioGuest11234567890.m4a", "teacher": "audioGuest11234567890.m4a",
      "confidence": 0.9, "reason": "x"}, "two roles"),
    ({"student": "nothere.m4a", "teacher": "audioiPhone21234567890.m4a",
      "confidence": 0.9, "reason": "x"}, "not there"),
])
def test_an_unsure_or_inconsistent_classifier_takes_the_mixed_path(tmp_path, clips, answer, why):
    sel = select(lesson(tmp_path, clips, **UNKNOWN), LEN, classify=lambda f, w: answer)
    assert sel.path == "mixed"
    assert why in sel.reason


def test_a_classifier_that_crashes_takes_the_mixed_path(tmp_path, clips):
    def classify(files, work):
        raise ConnectionError("network down")

    sel = select(lesson(tmp_path, clips, **UNKNOWN), LEN, classify=classify)
    assert sel.path == "mixed"
    assert "classifier failed" in sel.reason


def test_a_missing_mic_is_never_sent_to_the_classifier(tmp_path, clips):
    def classify(files, work):
        raise AssertionError("a voice that was not recorded cannot be classified into being")

    sel = select(lesson(tmp_path, clips, **{STEVEN: "steven", SHARE: "share"}), LEN, classify=classify)
    assert sel.path == "mixed"
    assert "mic track is missing" in sel.reason


def test_later_stages_read_the_saved_roles(tmp_path, clips):
    from jp_lesson_distill.tracks import selected_paths
    work = tmp_path / "work"
    select(lesson(tmp_path, clips, **RENUMBERED), LEN, work=work)
    paths = selected_paths(work)
    assert paths["student"].name == "audioSteveSaito21936486817.m4a"


def test_forcing_the_mixed_path_skips_the_tracks(tmp_path, clips):
    sel = select(lesson(tmp_path, clips, **FULL), LEN, mode="mixed")
    assert sel.path == "mixed"
    assert "--mixed" in sel.reason


def test_forcing_per_track_stops_with_the_reason_instead_of_falling_back(tmp_path, clips):
    with pytest.raises(SystemExit, match="no per-participant tracks"):
        select(lesson(tmp_path, clips), LEN, mode="per-track")


def test_the_track_folder_is_found_from_the_video_too(tmp_path, clips):
    rec = lesson(tmp_path, clips, **FULL)
    video = rec.with_name("Soso_20991025_class_video.mp4")
    assert tracks_dir(video) == tracks_dir(rec) == rec.with_suffix("")
