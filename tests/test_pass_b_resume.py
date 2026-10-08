"""Pass B keeps what it has paid for (jld-8rb).

On 2026-10-08 a run had re-listened 15 of 18 clips when clip 16's call failed four times;
moments.json is only written after the last clip, so all 15 were lost and a re-run began at
clip 1. Offline: the Gemini call and ffmpeg are replaced.
"""
import httpx
import pytest

from jp_lesson_distill import pipeline
from jp_lesson_distill.models import Candidate, CandidateList, MomentsFile, Relisten, Segment, Transcript

DATE = "20990101"


def lesson(tmp_path, monkeypatch, n=5):
    work = tmp_path / "work" / DATE
    work.mkdir(parents=True)
    (work / "audio.m4a").touch()
    pipeline._save(work / "transcript.json", Transcript(segments=[
        Segment(start="00:00", speaker="student", text="x", uncertain=False)]))
    pipeline._save(work / "candidates.json", CandidateList(candidates=[
        Candidate(t_start=f"0{i}:00", t_end=f"0{i}:10", type="correction", rationale="r",
                  excerpt="e", priority=1 - i / 10) for i in range(n)]))
    monkeypatch.setattr(pipeline, "duration_seconds", lambda _: 3600.0)
    monkeypatch.setattr(pipeline, "clip_audio", lambda audio, clip, a, b: clip.touch())
    monkeypatch.setattr(pipeline, "make_client", lambda *_: object())
    return work, pipeline.Config(recording=tmp_path / "rec.m4a", date=DATE, work_dir=tmp_path / "work",
                                 out_dir=tmp_path / "out", archive=False, board=False)


def relisten(fail_on=()):
    calls = []

    def fake(client, model, clip, cand):
        calls.append(clip.stem)
        if clip.stem in fail_on:
            raise httpx.ReadTimeout("the stream went quiet")
        return Relisten(keep=True, type="correction", student_verbatim=clip.stem,
                        teacher_correction="y", explanation="z", confidence=0.9)
    return fake, calls


def test_one_failed_clip_costs_one_clip(tmp_path, monkeypatch):
    work, cfg = lesson(tmp_path, monkeypatch)
    fake, calls = relisten(fail_on={"m03"})
    monkeypatch.setattr(pipeline, "_relisten", fake)

    with pytest.raises(SystemExit) as stop:
        pipeline.run(cfg)

    assert calls == ["m01", "m02", "m03", "m04", "m05"]  # it carried on past the failure
    assert sorted(p.stem for p in (work / "relisten").iterdir()) == ["m01", "m02", "m04", "m05"]
    # a moments.json missing one moment would look finished and be cached as the answer
    assert not (work / "moments.json").exists()
    assert "1 of 5" in str(stop.value) and "run the same command again" in str(stop.value)

    fake, calls = relisten()
    monkeypatch.setattr(pipeline, "_relisten", fake)
    pipeline.run(cfg)

    assert calls == ["m03"]  # only the gap is paid for again
    moments = pipeline._load(work / "moments.json", MomentsFile).moments
    assert [m.student_verbatim for m in moments] == ["m01", "m02", "m03", "m04", "m05"]


def test_two_failures_in_a_row_stop_the_rest_from_being_tried(tmp_path, monkeypatch):
    """A dead connection fails every clip the same way, four idle timeouts apiece."""
    work, cfg = lesson(tmp_path, monkeypatch)
    fake, calls = relisten(fail_on={"m02", "m03", "m04", "m05"})
    monkeypatch.setattr(pipeline, "_relisten", fake)

    with pytest.raises(SystemExit) as stop:
        pipeline.run(cfg)

    assert calls == ["m01", "m02", "m03"]
    assert str(stop.value).count("not tried") == 2
    assert [p.stem for p in (work / "relisten").iterdir()] == ["m01"]


def test_a_rejected_clip_is_remembered_as_rejected(tmp_path, monkeypatch):
    work, cfg = lesson(tmp_path, monkeypatch, n=2)

    def fake(client, model, clip, cand):
        return Relisten(keep=clip.stem == "m01", type="correction", student_verbatim=clip.stem,
                        explanation="z", confidence=0.9)
    monkeypatch.setattr(pipeline, "_relisten", fake)
    pipeline.run(cfg)

    assert [m.id for m in pipeline._load(work / "moments.json", MomentsFile).moments] == ["m01"]
    assert not pipeline._load(work / "relisten" / "m02.json", Relisten).keep
