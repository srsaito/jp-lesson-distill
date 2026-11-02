"""Pass B clips cover the student's whole last turn (jld-zxe).

On 10/26 one 182-character turn began at 09:31 with the error about 45 s in. Detect said
09:31-09:43, the clip stopped at 09:58, and Pass B rejected a real moment as cut off.
"""
from jp_lesson_distill.models import Segment, fmt_ts
from jp_lesson_distill.pipeline import CLIP_MAX_S, CLIP_PAD, clip_end

TOTAL = 3600.0


def student(t, chars):
    return Segment(start=fmt_ts(t), speaker="student", text="あ" * chars)


def teacher(t, chars=2):
    return Segment(start=fmt_ts(t), speaker="teacher", text="う" * chars)


def test_the_real_case_now_reaches_the_error():
    start, error_at = 9 * 60 + 31, 10 * 60 + 14
    end = clip_end([student(start, 182)], start, start + 12, TOTAL)
    assert end > error_at + 10, "the error and a little of what follows are in the clip"
    assert abs(end - (start + 182 / 3 + CLIP_PAD)) < 1


def test_a_short_turn_leaves_the_clip_as_it_was():
    assert clip_end([student(100, 20)], 100, 110, TOTAL) == 110 + CLIP_PAD


def test_the_clip_stops_at_his_next_segment():
    end = clip_end([student(100, 300), student(120, 10)], 100, 110, TOTAL)
    assert end == 120 + CLIP_PAD


def test_a_teacher_backchannel_mid_turn_does_not_end_it():
    with_uun = [student(100, 150), teacher(110), student(300, 10)]
    assert clip_end(with_uun, 100, 105, TOTAL) == 100 + 50 + CLIP_PAD


def test_the_turn_that_counts_is_his_last_before_detects_end():
    segs = [student(100, 300), student(150, 30), student(400, 300)]
    assert clip_end(segs, 100, 160, TOTAL) == 160 + CLIP_PAD


def test_growth_is_capped_but_detects_own_span_is_never_cut():
    assert clip_end([student(100, 3000)], 100, 110, TOTAL) == 100 - CLIP_PAD + CLIP_MAX_S
    assert clip_end([student(100, 3000)], 100, 400, TOTAL) == 400 + CLIP_PAD


def test_no_student_segment_near_changes_nothing():
    assert clip_end([teacher(100, 500)], 100, 110, TOTAL) == 110 + CLIP_PAD
    assert clip_end([student(500, 300)], 100, 110, TOTAL) == 110 + CLIP_PAD


def test_the_clip_stays_inside_the_recording():
    assert clip_end([student(100, 300)], 100, 110, 130.0) == 130.0
