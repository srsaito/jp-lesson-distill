"""Per-track Pass A (ADR-0010, jld-8nv.2), offline: Gemini is replaced by a stub.

Synthetic lesson: Steven's mic sounds from 1.0 to 2.5 s, Soso先生's from 3.0 to 5.0 s, digital
silence elsewhere. The stub answers like the model does, in whole seconds and a little off.
"""
import json

import numpy as np
import pytest

from jp_lesson_distill import pipeline
from jp_lesson_distill.models import Segment, TrackSegment, TrackTranscript, Transcript, parse_ts
from jp_lesson_distill.pipeline import Config
from jp_lesson_distill.tracks import FRAME_S, select, snap_start

from test_tracks import FULL, LEN, lesson  # noqa: F401 — synthetic tracks
from test_tracks import clips  # noqa: F401 — the fixture


def db_with(spans, total=10.0):
    db = np.full(int(total / FRAME_S), -120.0)
    for a, b in spans:
        db[int(a / FRAME_S):int(b / FRAME_S)] = -20.0
    return db


def test_a_start_inside_speech_snaps_back_to_where_the_voice_began():
    db = db_with([(3.0, 5.0)])
    assert snap_start(db, 4.0) == 3.0


def test_a_start_in_silence_snaps_to_the_nearest_onset_within_three_seconds():
    db = db_with([(3.0, 5.0), (8.0, 9.0)])
    assert snap_start(db, 2.0) == 3.0
    assert snap_start(db, 7.0) == 8.0


def test_a_start_with_no_sound_nearby_is_left_alone():
    assert snap_start(db_with([(9.0, 9.5)]), 2.0) == 2.0


def stub(monkeypatch, answers, fail=()):
    """Replace upload and generate; `answers` maps role -> segments, `fail` lists roles that raise."""
    monkeypatch.setattr(pipeline, "upload_audio", lambda client, path: path)

    def fake_generate(client, model, contents, schema, **kw):
        role = "teacher" if contents[1] == pipeline.TRACK_PROMPTS["teacher"] else "student"
        if role in fail:
            raise RuntimeError("stream went quiet")
        return TrackTranscript(segments=[TrackSegment(**s) for s in answers[role]])
    monkeypatch.setattr(pipeline, "generate", fake_generate)


def per_track_lesson(tmp_path, clips):
    rec = lesson(tmp_path, clips, **FULL)
    work = tmp_path / "work" / "20991025"
    work.mkdir(parents=True)
    audio = work / "audio.m4a"
    audio.write_bytes(clips["mix"].read_bytes())
    assert select(rec, LEN, work=work).path == "per-track"
    return Config(recording=rec, date="20991025", work_dir=tmp_path / "work"), work, audio


ANSWERS = {
    "student": [{"start": "00:00", "text": "えーと、学校で行きました。"}],
    "teacher": [{"start": "00:04", "text": "学校に行きました、ですね。"}],
}


def test_each_track_becomes_its_speaker_and_starts_land_on_the_voice(tmp_path, clips, monkeypatch):
    cfg, work, audio = per_track_lesson(tmp_path, clips)
    stub(monkeypatch, ANSWERS)
    t = pipeline._pass_a_tracks(cfg, work, audio, LEN, get_client=lambda: None)
    assert [(s.speaker, s.text) for s in t.segments] == [
        ("student", "えーと、学校で行きました。"),
        ("teacher", "学校に行きました、ですね。"),
    ]
    # Starts are written in whole seconds, and AAC blurs an onset by a frame or so, so "on the
    # voice" means within a second of it: the teacher's 00:04 moves back to where he began (3.0).
    student, teacher = (parse_ts(s.start) for s in t.segments)
    assert abs(student - 1.0) <= 1.0
    assert abs(teacher - 3.0) <= 1.0 and t.segments[1].start != "00:04"
    assert json.loads((work / "pass_a.json").read_text()) == {"path": "per-track", "fallback_windows": []}
    assert pipeline._pass_a_path(work) == "per-track"


def test_a_failed_window_falls_back_to_the_mix_and_its_labels_are_repaired(tmp_path, clips, monkeypatch):
    cfg, work, audio = per_track_lesson(tmp_path, clips)
    stub(monkeypatch, ANSWERS, fail={"student"})
    # the mixed transcription of the window, with the two speakers swapped (the 10/14 failure)
    swapped = Transcript(segments=[Segment(start="00:01", speaker="teacher", text="学校で行きました。"),
                                   Segment(start="00:03", speaker="student", text="学校に行きました。")])
    monkeypatch.setattr(pipeline, "_transcribe_window", lambda *a, **k: swapped)
    t = pipeline._pass_a_tracks(cfg, work, audio, LEN, get_client=lambda: None)
    assert [s.speaker for s in t.segments] == ["student", "teacher"], "repaired from the tracks"
    assert json.loads((work / "pass_a.json").read_text())["fallback_windows"] == [1]


def test_a_cached_transcript_without_a_record_is_treated_as_mixed(tmp_path):
    (tmp_path / "transcript.json").write_text(Transcript(segments=[]).model_dump_json())
    assert pipeline._pass_a_path(tmp_path) == "mixed"


def test_the_student_prompt_carries_no_teacher_context_and_both_keep_the_verbatim_rule():
    from jp_lesson_distill import prompts
    assert "CONTEXT" not in prompts.TRACK_STUDENT and "teacher said" not in prompts.TRACK_STUDENT
    for p in (prompts.TRACK_STUDENT, prompts.TRACK_TEACHER):
        assert "NEVER fix or smooth" in p
