"""Pass A sanity gates (jld-hc9.4).

Offline. The synthetic cases always run; the fixture cases use the transcripts under
work/ when they are present and skip otherwise. The fixture cases ARE the calibration:
the gates exist to separate the known-degraded 2026-08-17 single-pass run from the
known-good windows of the same lesson, and segments-per-minute demonstrably cannot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jp_lesson_distill.audio import duration_seconds
from jp_lesson_distill.models import Segment, Transcript
from jp_lesson_distill.quality import evaluate

WORK = Path(__file__).resolve().parents[1] / "work"


def seg(start: str, text: str, speaker: str = "teacher") -> Segment:
    return Segment(start=start, speaker=speaker, text=text)


def healthy(minutes: int = 20, per_min: int = 8) -> list[Segment]:
    """A plausible window: a turn every few seconds, running to the end."""
    out = []
    step = 60 // per_min
    for t in range(0, minutes * 60 - step, step):
        out.append(seg(f"{t // 60:02d}:{t % 60:02d}", "はい、そうですね。ありがとうございます。"))
    return out


# --- the gates that fire ---

def test_healthy_window_passes():
    report = evaluate(healthy(), 20 * 60)
    assert report.ok, [c.detail for c in report.failures]
    assert not report.warnings


def test_truncated_window_fails_on_coverage():
    segs = [s for s in healthy() if s.start < "17:00"]
    report = evaluate(segs, 20 * 60)
    assert not report.ok
    # Truncation trips coverage first, but the silent tail is also dead time and one
    # over-long span — the gates overlap on purpose; any one of them is enough.
    assert "coverage" in [c.name for c in report.failures]
    assert report.coverage < 0.9


def test_backwards_timestamps_fail_monotonic():
    segs = healthy()[:20]
    segs[10] = seg("00:05", "戻ってしまった。")
    report = evaluate(segs, 20 * 60)
    assert "monotonic" in [c.name for c in report.failures]


def test_a_minute_of_audio_behind_nine_characters_is_dead_time():
    # The real signature from the degraded run: 「うん。いいですね。」 covering 64 s.
    segs = healthy(minutes=20)
    segs = [s for s in segs if s.start < "05:00"] + [seg("05:00", "うん。いいですね。")] \
        + [s for s in segs if s.start >= "12:00"]
    report = evaluate(segs, 20 * 60)
    names = [c.name for c in report.failures]
    assert "dead-time" in names and "max-span" in names
    assert report.dead_time >= 7 * 60


def test_empty_window_fails_rather_than_passing_vacuously():
    report = evaluate([], 20 * 60)
    assert not report.ok


# --- warnings must not trigger a retry ---

def test_under_split_turn_warns_but_does_not_fail():
    segs = [s for s in healthy() if s.start < "10:00"]
    segs.append(seg("10:00", "あ" * 400))  # 90 s of continuous speech, one segment
    segs += [s for s in healthy() if s.start >= "11:30"]
    report = evaluate(segs, 20 * 60)
    assert report.ok
    assert "under-split" in [c.name for c in report.warnings]


def test_repeated_drill_line_warns_but_does_not_fail():
    segs = healthy()
    for i in (10, 40, 70):
        segs[i] = seg(segs[i].start, "木村さん、ネットでパソコン買ったことありますか？")
    report = evaluate(segs, 20 * 60)
    assert report.ok
    assert "duplicates" in [c.name for c in report.warnings]
    assert report.duplicates == 2


# --- calibration against the real transcripts ---

def _load(path: Path, audio: Path):
    if not path.exists() or not audio.exists():
        pytest.skip(f"fixture missing: {path}")
    return Transcript.model_validate_json(path.read_text()).segments, duration_seconds(audio)


@pytest.mark.parametrize("window", ["w1", "w2", "w3"])
def test_reference_windows_pass(window):
    d = WORK / "ref-windowed" / window / "20260817"
    segments, duration = _load(d / "transcript.json", d / "audio.m4a")
    report = evaluate(segments, duration)
    assert report.ok, f"{window}: {[c.detail for c in report.failures]}"


def test_windowed_run_passes():
    d = WORK / "hc9-2-verify/20260817"
    for i in (1, 2, 3):
        segments, duration = _load(d / f"transcript_w{i:02d}.json", d / "windows" / f"w{i:02d}.m4a")
        report = evaluate(segments, duration)
        assert report.ok, f"w{i}: {[c.detail for c in report.failures]}"


def test_the_degraded_single_pass_run_is_rejected():
    """hc9.4's acceptance criterion: feed the bad 20260817 transcript through as one window."""
    d = WORK / "20260817"
    segments, duration = _load(d / "transcript.json", d / "audio.m4a")
    report = evaluate(segments, duration)
    assert not report.ok
    names = [c.name for c in report.failures]
    assert "coverage" in names, "it stops at 49:59 of 56:00"
    assert "dead-time" in names, "eight minutes of audio carry almost no text"


def test_density_would_not_have_caught_it():
    """Why the gate is coverage/dead-time and not segments-per-minute."""
    bad_segments, bad_duration = _load(WORK / "20260817/transcript.json",
                                       WORK / "20260817/audio.m4a")
    good = WORK / "ref-windowed/w2/20260817"
    good_segments, good_duration = _load(good / "transcript.json", good / "audio.m4a")
    bad = evaluate(bad_segments, bad_duration)
    ok = evaluate(good_segments, good_duration)
    assert bad.density > ok.density, (
        f"the degraded run ({bad.density:.1f}/min) is DENSER than a good window "
        f"({ok.density:.1f}/min) — any density threshold separating them is backwards"
    )
