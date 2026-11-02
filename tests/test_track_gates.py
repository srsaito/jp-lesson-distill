"""Per-track quality gates (jld-8nv.3): judge a track's text against how much that person spoke.

The calibration fixture holds counts only, no utterance text (the repo is public): every per-track
window on disk from 10/12, 10/14, 10/25 and 10/26. The pipeline tests stub Gemini.
"""
import json
from pathlib import Path

import numpy as np

from jp_lesson_distill import pipeline, quality
from jp_lesson_distill.models import Segment, TrackSegment, TrackTranscript, Transcript
from jp_lesson_distill.quality import evaluate_track, is_ghost
from jp_lesson_distill.tracks import FRAME_S, SOUND_DB

from test_per_track import ANSWERS, per_track_lesson, stub
from test_tracks import FULL, LEN, STEVEN, make  # noqa: F401
from test_tracks import clips  # noqa: F401 — the fixture

ROWS = json.loads((Path(__file__).parent / "fixtures/track_gates_calibration.json").read_text())
DEFECTIVE = "20261026 w3 steven_ctx"  # 19 of 84 lines timed where the track is silent


def track(*spans, total=1200.0):
    db = np.full(int(total / FRAME_S), -120.0)
    for a, b in spans:
        db[int(a / FRAME_S):int(b / FRAME_S)] = -20.0
    return db


def judge(starts, texts, db):
    return evaluate_track(starts, texts, db, FRAME_S, SOUND_DB, 0.0, len(db) * FRAME_S)


def line(chars):
    return "あ" * chars


# 300 s of speech in 10 s bursts, one line of 40 chars each: 4 chars/s, like every real window
BURSTS = [(i * 20.0, i * 20.0 + 10.0) for i in range(30)]
STARTS = [a for a, _ in BURSTS]


def test_a_healthy_window_passes():
    r = judge(STARTS, [line(40)] * 30, track(*BURSTS))
    assert r.ok, [c.detail for c in r.failures]
    assert abs(r.rate - 4.0) < 0.1


def test_a_silently_compressed_window_is_caught():
    r = judge(STARTS[::3], [line(40)] * 10, track(*BURSTS))
    assert [c.name for c in r.failures] == ["text-missing"]


def test_text_beyond_what_was_said_is_caught():
    r = judge(STARTS, [line(120)] * 30, track(*BURSTS))
    assert "text-invented" in [c.name for c in r.failures]


def test_a_hole_is_caught_even_when_the_total_looks_fine():
    # the Gemini 3.5 Transcribe failure shape: normal text, then minutes of speech with none
    keep = [i for i in range(30) if not 5 <= i < 25]
    r = judge([STARTS[i] for i in keep], [line(120)] * len(keep), track(*BURSTS))
    assert "dropout" in [c.name for c in r.failures]


def test_a_long_lumped_segment_is_not_a_failure():
    # 10/26 w1: the same text in 27 segments or 78. Segmentation varies; only holes matter.
    r = judge(STARTS[::5], [line(200)] * 6, track(*BURSTS))
    assert r.ok, [c.detail for c in r.failures]


def test_lines_timed_on_silence_are_caught_as_a_timing_failure():
    db = track((0, 300))
    far = [600.0 + i * 20 for i in range(10)]
    r = judge(STARTS[:15] + far, [line(40)] * 25, db)
    assert "timing" in [c.name for c in r.failures]


def test_a_ghost_is_judged_near_its_own_start():
    db = track((10, 20))
    assert not is_ghost(db, FRAME_S, SOUND_DB, 9.0)    # the voice starts a second later
    assert not is_ghost(db, FRAME_S, SOUND_DB, 21.0)   # just after it stops
    assert is_ghost(db, FRAME_S, SOUND_DB, 40.0)


def test_a_short_window_is_not_judged_on_rate():
    # 10/26's two-minute tail: too little speech for a rate to mean anything
    r = judge([1.0], [line(300)], track((0, 10), total=120.0))
    assert r.ok


def test_the_thresholds_clear_every_healthy_window_on_disk():
    healthy = [r for r in ROWS if r["window"] != DEFECTIVE]
    assert len(healthy) == 37 and all(r["ok"] for r in healthy)
    judged = [r["chars"] / r["speech_s"] for r in healthy if r["speech_s"] >= quality.TRACK_MIN_SPEECH_S]
    assert min(judged) > 1.4 * quality.TRACK_RATE_FLOOR, "the floor stays well below healthy text"
    assert max(judged) * 1.5 < quality.TRACK_RATE_CEILING
    assert max(r["max_speech_gap_s"] for r in healthy) * 1.5 < quality.TRACK_MAX_SPEECH_GAP_S
    assert max(r["ghosts"] / r["segments"] for r in healthy) * 2 < quality.TRACK_GHOST_FRAC


def test_the_defective_window_is_the_one_rejected():
    bad = next(r for r in ROWS if r["window"] == DEFECTIVE)
    assert not bad["ok"] and bad["ghosts"] / bad["segments"] > quality.TRACK_GHOST_FRAC


def counting_stub(monkeypatch, answers, first_failures=0):
    calls = {"teacher": 0, "student": 0}
    monkeypatch.setattr(pipeline, "upload_audio", lambda client, path: path)

    def fake(client, model, contents, schema, **kw):
        role = "teacher" if contents[1] == pipeline.TRACK_PROMPTS["teacher"] else "student"
        calls[role] += 1
        if role == "student" and calls[role] <= first_failures:
            raise RuntimeError("stream went quiet")
        return TrackTranscript(segments=[TrackSegment(**s) for s in answers[role]])
    monkeypatch.setattr(pipeline, "generate", fake)
    return calls


def test_a_failed_call_is_retried_and_only_that_track_is_redone(tmp_path, clips, monkeypatch):
    cfg, work, audio = per_track_lesson(tmp_path, clips)
    calls = counting_stub(monkeypatch, ANSWERS, first_failures=1)
    t = pipeline._pass_a_tracks(cfg, work, audio, LEN, get_client=lambda: None)
    assert calls == {"teacher": 1, "student": 2}
    assert json.loads((work / "pass_a.json").read_text())["fallback_windows"] == []
    assert len(t.segments) == 2


def test_a_track_the_gates_keep_rejecting_sends_the_window_to_the_mix(tmp_path, clips, monkeypatch):
    cfg, work, audio = per_track_lesson(tmp_path, clips)
    calls = counting_stub(monkeypatch, ANSWERS)
    reject = quality.TrackReport(100.0, 1, 1, 0.0, 0,
                                 checks=[quality.Check("text-missing", False, "too little text")])
    monkeypatch.setattr(pipeline, "evaluate_track", lambda *a, **k: reject)
    mixed = Transcript(segments=[Segment(start="00:01", speaker="student", text="学校で行きました。"),
                                 Segment(start="00:03", speaker="teacher", text="学校に行きました。")])
    monkeypatch.setattr(pipeline, "_transcribe_window", lambda *a, **k: mixed)
    pipeline._pass_a_tracks(cfg, work, audio, LEN, get_client=lambda: None)
    assert calls["student"] == cfg.window_attempts
    assert json.loads((work / "pass_a.json").read_text())["fallback_windows"] == [1]
    assert (work / "tracks" / "student_w01.attempt1.json").exists(), "rejected attempts are kept"


def test_a_line_with_no_sound_nearby_is_dropped_before_snapping_can_save_it(tmp_path, clips, monkeypatch):
    cfg, work, audio = per_track_lesson(tmp_path, clips)
    # Steven's mic sounds from 1.0 to 2.5 s; at 6 s (the end) there has been nothing for 3.5 s
    answers = {"teacher": ANSWERS["teacher"],
               "student": ANSWERS["student"] + [{"start": "00:06", "text": "うん。"}]}
    stub(monkeypatch, answers)
    t = pipeline._pass_a_tracks(cfg, work, audio, LEN, get_client=lambda: None)
    assert "うん。" not in [s.text for s in t.segments]


def test_a_track_with_nobody_on_it_costs_no_call(tmp_path, clips, monkeypatch):
    rec_files = dict(FULL)
    short = tmp_path / "short.m4a"
    make(short, speech=((1.0, 1.4),))
    clips = {**clips, "barely": short}
    rec_files[STEVEN] = "barely"
    from jp_lesson_distill.tracks import select
    from test_tracks import lesson
    from jp_lesson_distill.pipeline import Config
    rec = lesson(tmp_path, clips, **rec_files)
    work = tmp_path / "work" / "20991025"
    work.mkdir(parents=True)
    audio = work / "audio.m4a"
    audio.write_bytes(clips["mix"].read_bytes())
    assert select(rec, LEN, work=work).path == "per-track"
    calls = counting_stub(monkeypatch, ANSWERS)
    t = pipeline._pass_a_tracks(Config(recording=rec, date="20991025", work_dir=tmp_path / "work"),
                                work, audio, LEN, get_client=lambda: None)
    assert calls == {"teacher": 1, "student": 0}
    assert [s.speaker for s in t.segments] == ["teacher"]
