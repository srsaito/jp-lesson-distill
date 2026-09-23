"""What the thinking-level arms actually measured (jld-hc9.1, ADR-0007).

`low` is the obvious setting for Pass A and it is the wrong one, so the evidence is pinned
here rather than left in a decision record nobody re-reads. The shape of the argument is the
same one `test_diarization_truth.py` makes: a future change to the default has to beat these
numbers, not a hunch about what transcription "is".

Self-contained — the fixture carries counts and timings only (public repo, no utterance text).
"""

from __future__ import annotations

import json
from pathlib import Path

from jp_lesson_distill.gemini import MAX_OUTPUT_TOKENS, PASS_A_THINKING_LEVEL, THINKING_WARN_SHARE

FIXTURE = Path(__file__).parent / "fixtures" / "thinking_level_20260917.json"
DATA = json.loads(FIXTURE.read_text())


def arms(level: str) -> list[dict]:
    return [r for r in DATA["runs"] if r["level"] == level]


def cue_rate(runs: list[dict]) -> tuple[int, int]:
    return sum(r["address_forms"][0] for r in runs), sum(r["address_forms"][1] for r in runs)


def test_low_gets_the_speaker_wrong_far_more_often_than_the_reference_run():
    """The finding that overturned the issue's premise, and the reason for the default."""
    agree, total = cue_rate(arms("low"))
    assert (agree, total) == (6, 15)
    ref = DATA["reference"]["address_forms"]
    assert ref == [5, 5]
    assert agree / total < 0.5 < ref[0] / ref[1]


def test_medium_reads_as_consistent_where_low_does_not():
    from jp_lesson_distill.quality import MARKER_WARN

    agree, total = cue_rate(arms("medium"))
    assert agree / total >= MARKER_WARN
    low_agree, low_total = cue_rate(arms("low"))
    assert low_agree / low_total < MARKER_WARN


def test_the_default_is_the_level_that_was_measured_best():
    assert PASS_A_THINKING_LEVEL.lower() == "medium"


def test_low_is_not_failing_because_it_lost_the_audio():
    """Every other gate passes at `low`, which is exactly why the canary had to exist:
    coverage, density and dead time cannot see a speaker label (jld-86b)."""
    for run in arms("low"):
        assert run["coverage"] == 1.0
        assert 4.6 <= run["density"] <= 9.6   # the known-good band from jld-hc9.4
        assert run["dead_s"] == 0


def test_low_is_not_failing_because_it_started_fixing_the_learner_s_japanese():
    """The other thing that could have explained it. Filler rate and student volume hold."""
    proxy = DATA["verbatim_proxy"]
    ref_rate, ref_chars = proxy["reference"][0]
    for rate, chars in proxy["low"] + proxy["medium"]:
        assert rate >= ref_rate * 0.9
        assert abs(chars - ref_chars) / ref_chars < 0.1


def test_thinking_at_the_chosen_level_stays_clear_of_the_budget_warning():
    """`medium` costs 14% of the output window; the warning fires at 25%."""
    thinking = arms("medium")[0]["thinking_tokens"]
    assert thinking / MAX_OUTPUT_TOKENS < THINKING_WARN_SHARE


def test_medium_is_cheaper_than_the_model_s_own_default():
    """So this is not "make Pass A think harder" — it is slightly less than the status quo."""
    assert arms("medium")[0]["thinking_tokens"] < DATA["model_default_probe"]["thinking_tokens"]


def test_the_first_byte_latencies_are_level_dependent_and_well_under_the_watchdog():
    """jld-hc9.3.3's calibration data: the idle timeout has to clear the SLOWEST healthy
    first byte, and how long a healthy call waits depends on how hard it was told to think."""
    from jp_lesson_distill.gemini import STREAM_IDLE_TIMEOUT_S

    slowest = max([r["first_byte_s"] for r in DATA["runs"]]
                  + [DATA["model_default_probe"]["first_byte_s"]])
    assert slowest == 71.1
    assert slowest < STREAM_IDLE_TIMEOUT_S
    assert max(r["first_byte_s"] for r in arms("low")) < 5   # `low` answers almost at once
