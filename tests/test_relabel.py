"""Checking and repairing the mixed transcript's speaker labels from the tracks (jld-ec6).

On 10/14 window 3 came back with teacher and student swapped, the pooled address-form check
said "consistent", and detect lost 10 of 22 moments. These pin the two halves of the fix: the
per-window address-form line, and the energy relabel that repaired 10/14 by hand.
"""
import numpy as np

from jp_lesson_distill.models import Segment, fmt_ts
from jp_lesson_distill.pipeline import window_bounds
from jp_lesson_distill.quality import marker_by_window
from jp_lesson_distill.tracks import FRAME_S, relabel, speaker_from_energy

LOUD, SILENT = -20.0, -120.0


def energy(total, student=(), teacher=(), share=()):
    """Per-frame dB arrays: LOUD inside the given (start, end) spans, digital silence elsewhere."""
    n = int(total / FRAME_S)
    out = {}
    for role, spans in (("student", student), ("teacher", teacher), ("share", share)):
        db = np.full(n, SILENT)
        for a, b in spans:
            db[int(a / FRAME_S):int(b / FRAME_S)] = LOUD
        out[role] = db
    return out


def seg(t, speaker, text="はい。"):
    return Segment(start=fmt_ts(t), speaker=speaker, text=text)


def test_the_tracks_name_the_speaker_only_when_one_clearly_dominates():
    e = energy(30, student=[(0, 10)], teacher=[(10, 20), (25, 30)], share=[(20, 25)])
    assert speaker_from_energy(e, 0, 10) == "student"
    assert speaker_from_energy(e, 10, 20) == "teacher"
    assert speaker_from_energy(e, 20, 25) == "played"
    assert speaker_from_energy(e, 5, 15) is None       # half each: two voices, can't say
    assert speaker_from_energy(e, 7, 7) is None        # empty span


def test_contradicted_labels_flip_and_everything_else_is_left_alone():
    e = energy(40, student=[(0, 10)], teacher=[(10, 20)], share=[(30, 40)])
    segs = [seg(0, "teacher"),   # the tracks say student -> flipped
            seg(10, "teacher"),  # agrees
            seg(15, "student"),  # 15-20 is teacher -> flipped
            seg(20, "student"),  # 20-30 silent on every track -> undecided, untouched
            seg(30, "teacher")]  # playback -> left for jld-lg6
    out, (check,) = relabel(segs, e, 40, [(0, 40)])
    assert [s.speaker for s in out] == ["student", "teacher", "teacher", "student", "teacher"]
    assert (check.agree, check.flipped, check.undecided, check.played) == (1, 2, 1, 1)
    assert segs[0].speaker == "teacher", "the input transcript is not mutated"


def test_an_inverted_window_shows_up_in_its_own_count():
    # window 1 right, window 2 swapped: pooled, it would read 50%; per window it reads 2/2 and 0/2
    e = energy(40, student=[(0, 5), (20, 25)], teacher=[(5, 10), (25, 30)])
    segs = [seg(0, "student"), seg(5, "teacher"), seg(10, "student", ""),
            seg(20, "teacher"), seg(25, "student"), seg(30, "student", "")]
    _, (w1, w2) = relabel(segs, e, 40, [(0, 20), (20, 40)])
    assert (w1.agree, w1.flipped) == (2, 0)
    assert (w2.agree, w2.flipped) == (0, 2)
    assert w2.agreement == 0.0


def test_window_bounds_follow_the_merge_midpoint_rule():
    # 10/14: 3395.6 s, 20-minute windows at 0 / 1170 / 2340 s with 30 s of overlap
    bounds = window_bounds(3395.6, 1200, 30)
    assert bounds == [(0.0, 1185.0), (1185.0, 2355.0), (2355.0, 3395.6)]
    assert window_bounds(600, 1200, 30) == [(0.0, 600)]  # one window: no windowing


def test_address_forms_are_reported_per_window_and_unchecked_windows_are_named():
    # 10/14's shape: every cue in window 1, so the pooled verdict said "consistent"
    w1 = [seg(10 + i, "teacher", "スティーブンさんはどうですか？") for i in range(5)]
    w2 = [seg(1300 + i, "student", "はい。") for i in range(5)]
    line = marker_by_window(w1 + w2, [(0, 1185), (1185, 2355)])
    assert "w01 5/5" in line and "w02 none" in line
    assert "window 2 unverified" in line


def test_an_inverted_window_with_cues_is_named():
    inverted = [seg(10 + i, "student", "スティーブンさんはどうですか？") for i in range(5)]
    line = marker_by_window(inverted, [(0, 1185)])
    assert "w01 0/5" in line and "window 1 looks inverted" in line
