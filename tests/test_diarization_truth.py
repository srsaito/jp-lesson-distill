"""What the ears actually said (jld-dli).

Steven labelled all 41 clips of the 2026-08-17 kit on 2026-08-25, blind. These tests
pin what that measurement found, because every later decision about diarization rests
on it: a repair pass has to beat these numbers on these clips, not on a hunch.

Self-contained — the fixture carries the labels, so nothing here needs work/ or audio.
"""

from __future__ import annotations

import json
from pathlib import Path

from jp_lesson_distill.labeling import GRADED, Item, mcnemar, score

FIXTURE = Path(__file__).parent / "fixtures" / "diarization_truth_20260817.json"


def truth_items() -> list[Item]:
    """The kit as Item objects. Text is not in the fixture (public repo) and is not needed."""
    data = json.loads(FIXTURE.read_text())
    return [
        Item(id=f"{n:03d}", stratum=r["stratum"], t_start=r["t_start"], text="",
             speaker=r["pass_a"], clip="", order=n,
             compare_speaker=r["reference"], truth=r["truth"])
        for n, r in enumerate(data["items"])
    ]


def test_the_whole_kit_was_labelled():
    items = truth_items()
    assert len(items) == 41
    assert all(i.truth for i in items), "an unlabelled clip would quietly shrink every denominator"


def test_windowed_pass_a_is_right_about_nine_times_in_ten():
    scores, _ = score(truth_items())
    by = {s.stratum: s for s in scores}
    assert by["ALL"].n == 36        # 41 minus the 5 whose text spans two turns
    assert by["ALL"].correct == 32  # 89%, 95% CI 75-96%
    assert by["random"].accuracy > 0.9, "the base rate over the lesson as a whole"


def test_only_one_error_is_actually_a_teacher_student_swap():
    """The headline: the diarization itself is nearly right. The losses are elsewhere."""
    _, confusion = score(truth_items())
    swaps = {k: v for k, v in confusion.items()
             if k.split(" -> ")[0] in ("teacher", "student")}
    assert sum(swaps.values()) == 1, confusion
    assert sum(v for k, v in confusion.items() if k.startswith("played")) == 3


def test_played_textbook_audio_is_the_larger_error_source():
    """jld-lg6: three clips are a recording nobody in the room spoke, all called teacher."""
    items = truth_items()
    played = [i for i in items if i.truth == "played"]
    assert len(played) == 3
    assert all(i.speaker == "teacher" for i in played), \
        "playback lands on the teacher because he is the one holding the speaker"


def test_one_segment_in_eight_runs_two_turns_together():
    """Not a diarization defect at all — Pass A merged two speakers into one segment."""
    items = truth_items()
    mixed = [i for i in items if i.truth == "mixed"]
    assert len(mixed) == 5
    assert len(mixed) / len(items) > len([i for i in items if i.truth in GRADED
                                          and i.speaker != i.truth]) / len(items)


def test_the_hand_staged_reference_run_is_the_wrong_one():
    """It disagreed with the windowed run on 16 graded clips and lost every single one.

    This is why work/ref-windowed/ is a word-accuracy reference only: its speaker labels
    are inverted in blocks. Under coin flips 0-for-16 has p = 3e-5.
    """
    items = truth_items()
    contested = [i for i in items if i.compare_speaker and i.truth in GRADED]
    assert len(contested) == 16
    assert sum(i.compare_speaker == i.truth for i in contested) == 0
    assert sum(i.speaker == i.truth for i in contested) == 14


def test_a_repair_pass_has_almost_no_room_to_win():
    """The bar any future repair has to clear, expressed as the test it would have to pass.

    Flipping every contested line — the most a cue-based repair could plausibly do — makes
    things dramatically worse, because the windowed run already had them right.
    """
    items = truth_items()
    flip = {"teacher": "student", "student": "teacher"}
    after = {i.id: flip.get(i.speaker, i.speaker) if i.compare_speaker else i.speaker
             for i in items}
    fixed, broken, p = mcnemar(items, {}, after)
    assert broken > fixed and p < 0.01, (fixed, broken, p)
