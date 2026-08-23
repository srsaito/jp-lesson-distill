"""Sanity gates on a Pass A window (jld-hc9.4).

A degraded transcription validates against the schema and reads plausibly — the
2026-08-17 single-pass run produced 394 well-formed segments while quietly
replaying an earlier eight minutes over real lesson audio and stopping at 49:59
of 56:00. Nothing downstream can detect that, so the check belongs here, right
after the window comes back and while retrying it is still cheap.

Thresholds are calibrated against every transcript on disk (see jld-hc9.4);
what separates good from degraded is coverage, dead time and span — NOT
segments-per-minute, which puts the known-bad run (7.0/min) inside the range of
the known-good ones (4.6-9.6/min). Density is reported, never gated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import Segment, fmt_ts, parse_ts

# --- calibrated thresholds ---
COVERAGE_SLACK = 90.0       # s: how far before the end the last segment may start
COVERAGE_SLACK_FRAC = 0.05  # or this fraction of the window, whichever is larger
DEAD_TIME_FRAC = 0.05       # fail above this fraction of the window carrying no speech
DEAD_SPAN_MIN = 20.0        # s: a span shorter than this is never "dead", it is a pause
DEAD_CHARS_PER_S = 1.0      # below this, the span is transcribing almost nothing
SPAN_FAIL = 180.0           # s: no single segment should have to cover this much audio
SPAN_WARN = 60.0            # s: the prompt asks for turns split at ~30 s
DENSITY_FLOOR = 3.0         # segments per audio-minute; a floor, not a quality bar
DUP_MIN_CHARS = 20          # ignore short interjections when looking for repeats


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True  # False = warn only, never worth spending a retry on

    @property
    def failed(self) -> bool:
        return not self.ok


@dataclass
class WindowReport:
    segments: int
    minutes: float
    density: float          # segments per audio-minute — logged, never gated
    coverage: float         # last segment start as a fraction of the window
    max_span: float         # longest stretch of audio one segment must cover
    dead_time: float        # seconds of audio inside spans that carry almost no text
    duplicates: int
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.failed and c.fatal]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.failed and not c.fatal]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        """One line, always printed, so degradation is visible in the run log."""
        return (f"{self.segments} segments, {self.density:.1f}/min, "
                f"coverage {self.coverage:.0%}, longest span {self.max_span:.0f}s, "
                f"dead {self.dead_time:.0f}s")


def _spans(segments: list[Segment], duration: float) -> list[tuple[float, Segment]]:
    """How much audio each segment is responsible for, including the tail."""
    starts = [parse_ts(s.start) for s in segments]
    spans = [(starts[i + 1] - starts[i], segments[i]) for i in range(len(segments) - 1)]
    spans.append((max(0.0, duration - starts[-1]), segments[-1]))
    return spans


def evaluate(segments: list[Segment], duration: float) -> WindowReport:
    """Measure one window's transcript and run the gates over it."""
    minutes = max(duration / 60, 1e-6)
    if not segments:
        empty = WindowReport(0, minutes, 0.0, 0.0, duration, duration, 0)
        empty.checks = [Check("empty", False, "no segments at all")]
        return empty

    starts = [parse_ts(s.start) for s in segments]
    spans = _spans(segments, duration)
    dead = sum(g for g, s in spans
               if g > DEAD_SPAN_MIN and len(s.text) / g < DEAD_CHARS_PER_S)
    long_spans = [(g, s) for g, s in spans if g > SPAN_WARN]
    worst_span, worst_seg = max(spans, key=lambda pair: pair[0])
    texts = Counter(s.text for s in segments if len(s.text) > DUP_MIN_CHARS)
    dupes = sum(n - 1 for n in texts.values() if n > 1)

    report = WindowReport(
        segments=len(segments),
        minutes=minutes,
        density=len(segments) / minutes,
        coverage=starts[-1] / duration if duration else 0.0,
        max_span=worst_span,
        dead_time=dead,
        duplicates=dupes,
    )

    slack = max(COVERAGE_SLACK, COVERAGE_SLACK_FRAC * duration)
    report.checks = [
        Check(
            "coverage",
            starts[-1] >= duration - slack,
            f"last segment starts at {fmt_ts(starts[-1])} of {fmt_ts(duration)} "
            f"(allowed to fall short by {slack:.0f}s)",
        ),
        Check(
            "monotonic",
            starts == sorted(starts),
            "segment starts must not go backwards"
            + (f"; first drop at {fmt_ts(_first_drop(starts))}" if starts != sorted(starts) else ""),
        ),
        Check(
            "dead-time",
            dead <= DEAD_TIME_FRAC * duration,
            f"{dead:.0f}s of audio ({dead / duration:.0%}) sits in spans transcribing "
            f"under {DEAD_CHARS_PER_S:g} char/s",
        ),
        Check(
            "max-span",
            worst_span <= SPAN_FAIL,
            f"one segment at {worst_seg.start} covers {worst_span:.0f}s "
            f"with {len(worst_seg.text)} chars",
        ),
        # --- warnings: real signal, but retrying the window is not the answer ---
        Check(
            "density-floor",
            report.density >= DENSITY_FLOOR,
            f"{report.density:.1f} segments/min is below the {DENSITY_FLOOR:g}/min floor",
            fatal=False,
        ),
        Check(
            "under-split",
            not long_spans,
            f"{len(long_spans)} segment(s) cover more than {SPAN_WARN:g}s "
            f"(longest {worst_span:.0f}s at {worst_seg.start}); Pass B clips ±15s "
            "around a start, so a long turn is poorly targeted",
            fatal=False,
        ),
        Check(
            "duplicates",
            dupes == 0,
            f"{dupes} repeated segment text(s) — a drill repetition is normal, "
            "a replayed block is not; check coverage and dead-time before believing it",
            fatal=False,
        ),
    ]
    return report


def _first_drop(starts: list[float]) -> float:
    for a, b in zip(starts, starts[1:]):
        if b < a:
            return b
    return 0.0
