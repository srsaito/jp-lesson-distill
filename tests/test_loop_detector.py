"""The repetition-loop detector (jld-hc9.3.1).

Everything here is offline. The real loop sample and the reference transcripts live
under work/ (gitignored); those tests skip when the fixtures are absent.

The detector is checked the way the stream uses it: `find_repetition_loop` looks at
the last LOOP_TAIL_CHARS of whatever it is given, so calling it on a PREFIX of the
response is exactly equivalent to calling it on the rolling tail at that point in
the stream. Walking every prefix therefore covers every possible chunk boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jp_lesson_distill.gemini import (
    LOOP_MIN_REPEATS,
    LOOP_UNIT_MAX,
    find_repetition_loop,
)

WORK = Path(__file__).resolve().parents[1] / "work"
LOOP_SAMPLE = WORK / "20260729" / "pass_a_partial.attempt1.json"


def first_fire(text: str, start: int = 0, stop: int | None = None) -> int | None:
    """The prefix length at which the detector would first abort the stream."""
    for i in range(max(start, 1), (len(text) if stop is None else stop) + 1):
        if find_repetition_loop(text[:i]) is not None:
            return i
    return None


# --- synthetic behaviour -------------------------------------------------------

def test_fires_at_the_calibrated_repeat_count():
    unit = "持っ、"
    assert find_repetition_loop("前置き" + unit * (LOOP_MIN_REPEATS - 1)) is None
    assert find_repetition_loop("前置き" + unit * LOOP_MIN_REPEATS) == unit


def test_reports_the_shortest_unit():
    # "ああ" repeated is really "あ" repeated; naming the shortest cycle is what a
    # reader needs to see in the abort message.
    assert find_repetition_loop("あ" * (LOOP_MIN_REPEATS * 2)) == "あ"


def test_only_the_tail_counts():
    """A loop that has stopped is not a loop — the model recovered and moved on.

    Note the recovery text has to VARY. Twenty identical sentences in a row would
    trip the detector too, and rightly so: that is the same failure with a longer
    unit, not speech.
    """
    recovered = "".join(f'{{"start": "3{i:1d}:0{i % 10}", "text": "その靴を持って来ました。"}},'
                        for i in range(40))
    assert find_repetition_loop("持っ、" * LOOP_MIN_REPEATS + recovered) is None


def test_ignores_a_cycle_longer_than_the_unit_limit():
    unit = "".join(chr(0x3042 + i % 40) for i in range(LOOP_UNIT_MAX + 1))
    assert find_repetition_loop(unit * LOOP_MIN_REPEATS) is None


def test_short_input_never_fires():
    assert find_repetition_loop("") is None
    assert find_repetition_loop("はい。") is None


# --- real speech must stay quiet ----------------------------------------------

def test_genuine_backchannels_do_not_fire():
    """The measured worst case in 29 transcripts is five repeats, all backchannels."""
    for unit in ("そう", "はい", "うん", "ええ"):
        line = f'"text": "{unit * 5}、なるほどですね。"'
        assert find_repetition_loop(line) is None, unit


def test_json_structure_does_not_fire():
    """Indentation and punctuation repeat constantly in a streamed JSON body."""
    body = json.dumps(
        {"segments": [{"start": f"{m:02d}:{s:02d}", "speaker": "teacher", "text": "はい。"}
                      for m in range(20) for s in range(0, 60, 7)]},
        ensure_ascii=False, indent=2,
    )
    assert first_fire(body) is None


# --- the one real loop on disk ------------------------------------------------

def _loop_sample() -> str:
    if not LOOP_SAMPLE.exists():
        pytest.skip(f"fixture missing: {LOOP_SAMPLE}")
    return LOOP_SAMPLE.read_text()


def test_fires_on_the_20260729_loop_well_inside_the_budget():
    """Acceptance: abort within ~2k characters of the garbage starting.

    The real call emitted 58,356 characters of 「持っ、」 before the output budget
    ran out. Catching it in the first couple of hundred is the whole saving.
    """
    raw = _loop_sample()
    loop_start = raw.index("持っ、" * 6)
    fired = first_fire(raw, start=loop_start - 50, stop=loop_start + 2000)
    assert fired is not None, "the detector missed the only real loop we have"
    assert fired - loop_start < 2000
    # 20 repeats of a 3-character unit: it should be far tighter than the budget.
    assert fired - loop_start <= 4 * LOOP_MIN_REPEATS


def test_does_not_fire_on_the_healthy_prefix_of_that_same_call():
    """35 minutes of correct transcription precede the loop; none of it may trip."""
    raw = _loop_sample()
    loop_start = raw.index("持っ、" * 6)
    assert first_fire(raw, stop=loop_start) is None


# --- accepted transcripts must never fire -------------------------------------

REFERENCE = [
    WORK / "ref-windowed" / w / "20260817" / "transcript.json" for w in ("w1", "w2", "w3")
] + [
    WORK / "hc9-2-verify" / "20260817" / "transcript.json",
]


@pytest.mark.parametrize("path", REFERENCE, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_no_accepted_transcript_trips_the_detector(path: Path):
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    assert first_fire(path.read_text()) is None
