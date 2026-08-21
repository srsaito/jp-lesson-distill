"""Windowed Pass A: window planning, offset shifting, overlap merge, seam de-dup.

Everything here runs offline. The reference fixtures under work/ref-windowed/ are
known-good per-window transcripts (gitignored, local only); tests that need them
skip when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jp_lesson_distill.audio import window_starts
from jp_lesson_distill.models import Segment, Transcript, fmt_ts, parse_ts
from jp_lesson_distill.pipeline import merge_windows

REF = Path(__file__).resolve().parents[1] / "work" / "ref-windowed"
REF_DATE = "20260817"
REF_TOTAL = 3360.23  # duration of the 2026-08-17 recording, 56:00


def seg(start: str, speaker: str, text: str) -> Segment:
    return Segment(start=start, speaker=speaker, text=text)


# --- window planning ---

def test_no_windowing_when_recording_fits_in_one_window():
    assert window_starts(total=900, win_s=1200, overlap_s=30) == []
    assert window_starts(total=1200, win_s=1200, overlap_s=30) == []


def test_window_minutes_zero_means_single_pass():
    assert window_starts(total=3600, win_s=0, overlap_s=30) == []


def test_windows_stride_by_window_minus_overlap_and_cover_the_recording():
    starts = window_starts(total=3360.23, win_s=1200, overlap_s=30)
    assert starts == [0.0, 1170.0, 2340.0]
    assert starts[-1] + 1200 >= 3360.23  # last window runs to the end
    for a, b in zip(starts, starts[1:]):
        assert a + 1200 - b == pytest.approx(30)  # every seam overlaps by exactly 30 s


def test_trailing_sliver_is_folded_into_the_previous_window():
    starts = window_starts(total=1210, win_s=600, overlap_s=30)
    assert starts == [0.0, 570.0, 1140.0]
    # 19:20 of audio: a third window would hold 20 s the second already covers.
    starts = window_starts(total=1160, win_s=600, overlap_s=30)
    assert starts == [0.0, 570.0]
    assert 570 + 600 >= 1160


# --- merge ---

def test_offsets_shift_segments_to_absolute_time():
    parts = [[seg("00:10", "teacher", "A")], [seg("05:00", "student", "B")]]
    merged = merge_windows(parts, [0.0, 1170.0], win_s=1200, total=2400)
    assert [s.start for s in merged.segments] == ["00:10", "24:30"]


def test_overlap_is_split_at_its_midpoint():
    # Window 1 = 0-1200, window 2 = 1170-2370; overlap 1170-1200, midpoint 1185.
    parts = [
        [seg("19:31", "teacher", "before midpoint"), seg("19:50", "teacher", "w1 past midpoint")],
        [seg("00:05", "student", "w2 before midpoint"), seg("00:25", "student", "after midpoint")],
    ]
    merged = merge_windows(parts, [0.0, 1170.0], win_s=1200, total=2370)
    assert [s.text for s in merged.segments] == ["before midpoint", "after midpoint"]


def test_seam_duplicates_are_dropped_even_when_they_straddle_the_midpoint():
    # Both windows heard the same line, timed 4 s apart across the midpoint.
    parts = [
        [seg("19:41", "student", "面白いでした。")],
        [seg("00:15", "student", "面白いでした")],  # 1185 abs — same line, different punctuation
    ]
    merged = merge_windows(parts, [0.0, 1170.0], win_s=1200, total=2370)
    assert [s.text for s in merged.segments] == ["面白いでした。"]


def test_segments_past_the_end_of_the_recording_are_dropped():
    parts = [[seg("00:10", "teacher", "real"), seg("25:00", "teacher", "hallucinated")]]
    merged = merge_windows(parts, [0.0], win_s=0, total=900)
    assert [s.text for s in merged.segments] == ["real"]


def test_merged_output_is_sorted_and_monotonic():
    parts = [
        [seg("05:00", "teacher", "A"), seg("02:00", "student", "B")],  # model emitted out of order
        [seg("00:10", "teacher", "C")],
    ]
    merged = merge_windows(parts, [0.0, 1170.0], win_s=1200, total=2370)
    starts = [parse_ts(s.start) for s in merged.segments]
    assert starts == sorted(starts)


# --- against the real per-window reference transcripts ---

def _reference_windows() -> list[tuple[list[Segment], float]]:
    windows = []
    for name in ("w1", "w2", "w3"):
        d = REF / name / REF_DATE
        if not (d / "transcript.json").exists():
            pytest.skip(f"reference fixture missing: {d}")
        segments = Transcript.model_validate_json((d / "transcript.json").read_text()).segments
        windows.append((segments, float((d / "offset.txt").read_text().strip())))
    return windows


def test_reference_windows_merge_into_a_full_lesson():
    windows = _reference_windows()
    # The reference windows were cut at 0/1140/2340 and are 1170/1200/1020 s long;
    # win_s=1200 reproduces their seams.
    merged = merge_windows([w[0] for w in windows], [w[1] for w in windows],
                           win_s=1200, total=REF_TOTAL)

    starts = [parse_ts(s.start) for s in merged.segments]
    assert starts == sorted(starts), "merged timestamps must be monotonic"
    assert starts[0] <= 5, "merged transcript must start at the beginning of the recording"
    assert REF_TOTAL - starts[-1] < 120, "merged transcript must run to the end of the recording"
    assert len(merged.segments) >= 350, f"suspiciously few segments: {len(merged.segments)}"

    # No gap where a seam is: every 5-minute stretch of the lesson has speech in it.
    buckets = {int(s // 300) for s in starts}
    assert buckets == set(range(int(REF_TOTAL // 300) + 1)), f"empty stretch at buckets {buckets}"


def test_no_duplicate_segments_across_overlaps():
    windows = _reference_windows()
    merged = merge_windows([w[0] for w in windows], [w[1] for w in windows],
                           win_s=1200, total=REF_TOTAL)
    # A window's own seam is 30 s wide, so the same line by the same speaker inside
    # 30 s is a merge artifact. (The lesson does genuinely repeat lines — the same
    # roleplay line comes back 57 s later — hence the tight window.)
    seen: list[tuple[float, str, str]] = []
    for s in merged.segments:
        start = parse_ts(s.start)
        assert not [
            1 for prev, speaker, text in seen
            if start - prev <= 30 and speaker == s.speaker and text == s.text and len(text) > 6
        ], f"duplicate near {s.start}: {s.text}"
        seen.append((start, s.speaker, s.text))


def test_merged_content_matches_each_window_at_absolute_time():
    """Every merged segment is some window's segment, at that window's absolute time."""
    windows = _reference_windows()
    merged = merge_windows([w[0] for w in windows], [w[1] for w in windows],
                           win_s=1200, total=REF_TOTAL)
    source = {
        (fmt_ts(parse_ts(s.start) + offset), s.speaker, s.text)
        for segments, offset in windows for s in segments
    }
    for s in merged.segments:
        assert (s.start, s.speaker, s.text) in source, f"merged segment not in any window: {s}"
