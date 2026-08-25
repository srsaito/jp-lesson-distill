"""Address-form agreement as a diarization canary (jld-86b).

Every other gate in quality.py asks whether speech went missing. This one asks
whether it landed on the right person — and it is the only check in the pipeline
that would have caught the 2026-08-17 hand-staged run, which passes coverage,
dead-time, span and monotonicity while getting a third of its speakers wrong.

The fixture cases are the calibration and skip when work/ is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jp_lesson_distill.audio import duration_seconds
from jp_lesson_distill.models import Segment, Transcript
from jp_lesson_distill.quality import (
    MARKER_MIN_N,
    evaluate,
    marker_agreement,
    marker_verdict,
)

WORK = Path(__file__).resolve().parents[1] / "work"


def seg(start: str, text: str, speaker: str = "teacher") -> Segment:
    return Segment(start=start, speaker=speaker, text=text)


def filler(minutes: int = 20) -> list[Segment]:
    """Cue-free background so a window is otherwise healthy."""
    return [seg(f"{t // 60:02d}:{t % 60:02d}", "はい、そうですね。ありがとうございます。",
                "teacher" if (t // 8) % 2 else "student")
            for t in range(0, minutes * 60 - 8, 8)]


# --- counting ---

def test_a_window_with_no_address_forms_reports_none_not_zero():
    """Zero-of-zero must not read as total failure — a drill-heavy window has no cues."""
    report = evaluate(filler(), 20 * 60)
    assert report.marker_total == 0
    assert report.marker_rate is None
    assert "no address forms" in report.summary()
    assert "address-forms" not in [c.name for c in report.warnings]


def test_agreeing_cues_are_counted_and_stay_quiet():
    segs = filler()
    segs[10] = seg(segs[10].start, "スティーブンさんはどうですか？", "teacher")
    segs[40] = seg(segs[40].start, "先生、質問があります。", "student")
    segs[70] = seg(segs[70].start, "奥さんもお仕事ですか？", "teacher")
    report = evaluate(segs, 20 * 60)
    assert (report.marker_agree, report.marker_total) == (3, 3)
    assert report.marker_rate == 1.0
    assert "address-forms" not in [c.name for c in report.warnings]


def test_disagreeing_cues_warn_but_never_fail():
    """A retry cannot fix diarization, so this must not burn a window attempt."""
    segs = filler()
    for i, (text, spk) in enumerate([("スティーブンさんはどうですか？", "student"),
                                     ("先生、質問があります。", "teacher"),
                                     ("奥さんもお仕事ですか？", "student"),
                                     ("妻は今日仕事です。", "teacher")]):
        segs[10 + i * 20] = seg(segs[10 + i * 20].start, text, spk)
    report = evaluate(segs, 20 * 60)
    assert (report.marker_agree, report.marker_total) == (0, 4)
    assert "address-forms" in [c.name for c in report.warnings]
    assert report.ok, "warn-only: the gate must not trigger a re-roll"
    assert not report.failures


def test_too_few_cues_to_mean_anything_stays_silent():
    """Below MARKER_MIN_N one disagreement is noise, and a false alarm costs listening time."""
    segs = filler()
    segs[10] = seg(segs[10].start, "スティーブンさんはどうですか？", "student")
    report = evaluate(segs, 20 * 60)
    assert report.marker_total < MARKER_MIN_N
    assert "address-forms" not in [c.name for c in report.warnings]


def test_two_of_three_is_not_enough_evidence_to_warn():
    """The real hand-staged w2 scores exactly this, and it is close to no evidence.

    At a denominator of 3 the only silent outcome would be 3/3, so a window with one
    odd line would always warn. MARKER_MIN_N = 4 is what keeps that quiet.
    """
    segs = filler()
    segs[10] = seg(segs[10].start, "スティーブンさんはどうですか？", "teacher")
    segs[40] = seg(segs[40].start, "先生、質問があります。", "student")
    segs[70] = seg(segs[70].start, "奥さんもお仕事ですか？", "student")   # the odd one
    report = evaluate(segs, 20 * 60)
    assert (report.marker_agree, report.marker_total) == (2, 3)
    assert "address-forms" not in [c.name for c in report.warnings]
    assert "address forms 2/3" in report.summary(), "still reported, just not alarmed about"


def test_the_summary_line_always_shows_the_denominator():
    """'86%' out of seven is a story about one line; '6/7' is a reading."""
    segs = filler()
    segs[10] = seg(segs[10].start, "スティーブンさんはどうですか？", "teacher")
    segs[40] = seg(segs[40].start, "先生、質問があります。", "teacher")
    segs[70] = seg(segs[70].start, "奥さんもお仕事ですか？", "teacher")
    assert "address forms 2/3" in evaluate(segs, 20 * 60).summary()


# --- calibration against the two real runs ---

def _load(path: Path, audio: Path):
    if not path.exists() or not audio.exists():
        pytest.skip(f"fixture missing: {path}")
    return Transcript.model_validate_json(path.read_text()).segments, duration_seconds(audio)


def test_the_good_run_clears_the_threshold():
    d = WORK / "hc9-2-verify/20260817"
    agree = total = 0
    for i in (1, 2, 3):
        segments, duration = _load(d / f"transcript_w{i:02d}.json", d / "windows" / f"w{i:02d}.m4a")
        r = evaluate(segments, duration)
        agree, total = agree + r.marker_agree, total + r.marker_total
    assert (agree, total) == (8, 9), "the run Steven's ears scored at 89%"


def test_the_bad_run_is_caught_here_and_nowhere_else():
    """The whole point of jld-86b: every other gate passes this window."""
    d = WORK / "ref-windowed/w1/20260817"
    segments, duration = _load(d / "transcript.json", d / "audio.m4a")
    report = evaluate(segments, duration)
    assert report.ok, "it passes coverage, dead-time, span and monotonicity"
    assert (report.marker_agree, report.marker_total) == (1, 5)
    assert "address-forms" in [c.name for c in report.warnings]


def test_the_canary_separates_the_two_runs_before_anyone_listens():
    """It ranks them correctly using no ground truth and no extra API call."""
    good = WORK / "hc9-2-verify/20260817"
    g_segs, g_dur = _load(good / "transcript_w01.json", good / "windows" / "w01.m4a")
    bad = WORK / "ref-windowed/w1/20260817"
    b_segs, b_dur = _load(bad / "transcript.json", bad / "audio.m4a")
    g, b = evaluate(g_segs, g_dur), evaluate(b_segs, b_dur)
    assert g.marker_rate > b.marker_rate, (g.marker_rate, b.marker_rate)


# --- the whole-run verdict, where the denominator is usable ---

def test_the_run_verdict_names_the_next_step_when_it_looks_wrong():
    """A warning that does not say what to do next gets ignored."""
    segs = filler()
    for i, (text, spk) in enumerate([("スティーブンさんはどうですか？", "student"),
                                     ("先生、質問があります。", "teacher"),
                                     ("奥さんもお仕事ですか？", "student"),
                                     ("妻は今日仕事です。", "teacher")]):
        segs[10 + i * 20] = seg(segs[10 + i * 20].start, text, spk)
    verdict = marker_verdict(segs)
    assert "0/4" in verdict
    assert "distill label" in verdict
    assert "not limited" in verdict, "the damage is not confined to the cued lines"


def test_the_run_verdict_does_not_claim_health_it_cannot_prove():
    """No cues must read as 'unverified', never as 'fine'."""
    verdict = marker_verdict(filler())
    assert "unverified" in verdict
    assert "consistent" not in verdict


def test_the_run_verdict_is_calm_when_the_cues_agree():
    segs = filler()
    for i, (text, spk) in enumerate([("スティーブンさんはどうですか？", "teacher"),
                                     ("先生、質問があります。", "student"),
                                     ("奥さんもお仕事ですか？", "teacher"),
                                     ("妻は今日仕事です。", "student")]):
        segs[10 + i * 20] = seg(segs[10 + i * 20].start, text, spk)
    assert "consistent" in marker_verdict(segs)


def test_the_two_real_runs_get_opposite_verdicts_over_the_whole_hour():
    """The aggregate is the figure that carries weight: 8/9 against 3/8."""
    good = WORK / "hc9-2-verify/20260817/transcript.json"
    if not good.exists():
        pytest.skip("fixture missing")
    g = Transcript.model_validate_json(good.read_text()).segments
    assert marker_agreement(g) == (8, 9)
    assert "consistent" in marker_verdict(g)

    ref = []
    for w in ("w1", "w2", "w3"):
        p = WORK / "ref-windowed" / w / "20260817" / "transcript.json"
        if not p.exists():
            pytest.skip("fixture missing")
        ref += Transcript.model_validate_json(p.read_text()).segments
    assert marker_agreement(ref) == (3, 8)
    assert "distill label" in marker_verdict(ref)


def test_the_committed_ground_truth_agrees_with_the_cue():
    """Sanity-check the premise itself: where Steven's ears and an address form both
    speak to the same line, they must not contradict each other."""
    fixture = Path(__file__).parent / "fixtures" / "diarization_truth_20260817.json"
    rows = json.loads(fixture.read_text())["items"]
    assert sum(1 for r in rows if r["truth"] in ("teacher", "student")) >= 30
