"""Detect runs more than once and keeps the union (jld-dik). Gemini is stubbed."""
import httpx
import pytest

from jp_lesson_distill import pipeline
from jp_lesson_distill.models import Candidate, CandidateList, Segment, Transcript
from jp_lesson_distill.pipeline import Config, _detect_runs, merge_candidates, same_candidate

A = "学校で行きました。学校に行きました、ですね。"
B = "面白いでした。アクション映画です。"


def cand(start, end, excerpt, priority=0.5):
    return Candidate(t_start=start, t_end=end, type="correction", rationale="r", excerpt=excerpt, priority=priority)


def test_the_same_words_at_the_same_time_are_one_moment():
    assert same_candidate(cand("10:00", "10:10", A), cand("10:05", "10:15", A))


def test_different_errors_close_in_time_stay_separate():
    # the reason time alone is not the rule: a dense stretch holds several errors
    assert not same_candidate(cand("10:00", "10:10", A), cand("10:05", "10:15", B))


def test_the_same_words_much_later_are_a_new_moment():
    # a drill repeats a line; each pass is its own chance to make the error
    assert not same_candidate(cand("10:00", "10:10", A), cand("20:00", "20:10", A))


def test_a_repeat_widens_the_moment_and_lifts_its_priority():
    (m,) = merge_candidates([[cand("10:00", "10:10", A, 0.4)], [cand("09:55", "10:20", A, 0.9)]])
    assert (m.t_start, m.t_end, m.priority) == ("09:55", "10:20", 0.9)


def test_the_merge_keeps_what_only_one_run_found():
    merged = merge_candidates([[cand("01:00", "01:10", A)], [cand("30:00", "30:10", B)]])
    assert [m.t_start for m in merged] == ["01:00", "30:00"]


def test_one_run_never_merges_its_own_candidates():
    merged = merge_candidates([[cand("10:00", "10:10", A), cand("10:02", "10:12", A)]])
    assert len(merged) == 2


def test_times_compare_as_times_past_the_hour():
    (m,) = merge_candidates([[cand("59:55", "1:00:05", A)], [cand("1:00:00", "1:00:10", A)]])
    assert (m.t_start, m.t_end) == ("59:55", "1:00:10")


TRANSCRIPT = Transcript(segments=[Segment(start="00:00", speaker="student", text="x")])


def setup(tmp_path, monkeypatch, answers):
    calls = iter(answers)

    def fake(*a, **k):
        got = next(calls)
        if isinstance(got, Exception):
            raise got
        return CandidateList(candidates=got)
    monkeypatch.setattr(pipeline, "generate", fake)
    return Config(recording=tmp_path / "r.m4a", date="20990101")


def test_runs_are_each_cached(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch, [[cand("01:00", "01:10", A)], [cand("30:00", "30:10", B)]])
    cfg.detect_runs = 2
    assert len(_detect_runs(cfg, tmp_path, TRANSCRIPT, None).candidates) == 2
    assert (tmp_path / "candidates_run1.json").exists() and (tmp_path / "candidates_run2.json").exists()


def test_a_cached_run_is_not_paid_for_twice(tmp_path, monkeypatch):
    pipeline._save(tmp_path / "candidates_run1.json", CandidateList(candidates=[cand("01:00", "01:10", A)]))
    cfg = setup(tmp_path, monkeypatch, [[cand("30:00", "30:10", B)]])  # only one answer available
    assert len(_detect_runs(cfg, tmp_path, TRANSCRIPT, None).candidates) == 2


def test_a_failed_second_run_costs_only_its_extra_moments(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch, [[cand("01:00", "01:10", A)], httpx.ReadTimeout("quiet")])
    assert len(_detect_runs(cfg, tmp_path, TRANSCRIPT, None).candidates) == 1


def test_a_failed_first_run_still_fails(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch, [httpx.ReadTimeout("quiet")])
    with pytest.raises(httpx.ReadTimeout):
        _detect_runs(cfg, tmp_path, TRANSCRIPT, None)


def test_one_run_is_the_old_behaviour(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch, [[cand("01:00", "01:10", A)]])
    cfg.detect_runs = 1
    assert len(_detect_runs(cfg, tmp_path, TRANSCRIPT, None).candidates) == 1
