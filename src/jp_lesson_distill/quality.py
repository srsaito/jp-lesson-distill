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

These gates all ask "is any speech missing?". Address-form agreement (jld-86b)
asks the different question "is the speech attributed to the right person?" —
the one thing here that would have caught the 2026-08-17 hand-staged run, which
passes every gate above and still gets a third of its speakers wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .labeling import MARKERS, marker_conflict
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

# Address forms pin a speaker regardless of voice (jld-86b). This warns and never
# fails: a retry cannot fix diarization, so a failure here would only burn attempts.
#
# The cue is SPARSE — roughly nine lines an hour, so about three per 20-minute window,
# and a drill-heavy window can have none. That is why MARKER_MIN_N exists. At a
# denominator of 3 the only silent outcome is 3/3, so every window with a single
# disagreement would warn; on the real transcripts that fires on hand-staged w2 at
# 2/3, which is close to no evidence at all. Four is the smallest denominator where
# the check can distinguish "one odd line" from "this window is wrong": it keeps the
# good run's 3/4 window quiet and still catches the bad run's 1/5.
#
# The figure that actually carries weight is the WHOLE-RUN one (marker_agreement over
# the merged transcript), where the hour aggregates: 8/9 for the good 2026-08-17 run
# against 3/8 for the hand-staged one.
MARKER_WARN = 0.7           # agreement below this is worth cutting a listening kit
MARKER_MIN_N = 4            # fewer marker lines than this says nothing either way


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
    marker_agree: int = 0   # segments whose label matches the address form they contain
    marker_total: int = 0   # segments containing an address form at all
    checks: list[Check] = field(default_factory=list)

    @property
    def marker_rate(self) -> float | None:
        """None when the window has no address forms — not zero, which reads as failure."""
        return self.marker_agree / self.marker_total if self.marker_total else None

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
        # The address-form count is always shown with its denominator: "6/7" is a
        # reading, "86%" out of seven is a story about one line.
        cue = (f", address forms {self.marker_agree}/{self.marker_total}"
               if self.marker_total else ", no address forms")
        return (f"{self.segments} segments, {self.density:.1f}/min, "
                f"coverage {self.coverage:.0%}, longest span {self.max_span:.0f}s, "
                f"dead {self.dead_time:.0f}s{cue}")


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
    agree, n_cued = marker_agreement(segments)

    report = WindowReport(
        segments=len(segments),
        minutes=minutes,
        density=len(segments) / minutes,
        coverage=starts[-1] / duration if duration else 0.0,
        max_span=worst_span,
        dead_time=dead,
        duplicates=dupes,
        marker_agree=agree,
        marker_total=n_cued,
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
        Check(
            "address-forms",
            # Silent unless there are enough cues to mean anything. Every other gate
            # here asks whether speech went missing; this one asks whether it landed
            # on the right person, and nothing else in the pipeline does.
            n_cued < MARKER_MIN_N or agree / n_cued >= MARKER_WARN,
            f"only {agree}/{n_cued} segments containing 「スティーブンさん」「奥さん」"
            f"「先生」「妻」 are labelled the way the address form requires — diarization "
            "may be wrong well beyond these lines; cut a listening kit with `distill label`",
            fatal=False,
        ),
    ]
    return report


def _has_marker(segment: Segment) -> bool:
    """Whether the line contains an address form at all — conflict or not.

    marker_conflict() returns None both for 'agrees' and for 'no cue present', so the
    denominator has to be counted separately. MARKERS comes from labeling rather than
    being restated here: the gate and the ground-truth kit must never drift apart.
    """
    return any(marker in segment.text for marker, _ in MARKERS)


def marker_agreement(segments: list[Segment]) -> tuple[int, int]:
    """(agreeing, total) segments carrying an address form.

    Over a whole lesson this is the figure worth reading — an hour supplies roughly
    nine cues, enough to tell 8/9 from 3/8, where any single 20-minute window supplies
    about three and can only tell you that something might be off.
    """
    cued = [s for s in segments if _has_marker(s)]
    return sum(1 for s in cued if marker_conflict(s) is None), len(cued)


def marker_verdict(segments: list[Segment]) -> str:
    """One line about the whole run's diarization, for the end of Pass A.

    Deliberately never alarming on thin evidence: with no cues, or too few, it says so
    rather than implying a clean bill of health it cannot give.
    """
    agree, total = marker_agreement(segments)
    if not total:
        return ("no 「スティーブンさん」「奥さん」「先生」「妻」 lines in this lesson, so "
                "diarization is unverified — normal for a drill-heavy hour")
    head = f"address forms {agree}/{total}"
    if total < MARKER_MIN_N:
        return f"{head} — too few cues to judge diarization either way"
    if agree / total < MARKER_WARN:
        return (f"{head} — diarization looks unreliable, and the damage is not limited "
                f"to these {total} lines. Cut a listening kit: `distill label <recording> "
                "--date <date>` then `--play`, and `distill score` to measure it")
    return f"{head} — diarization consistent with the address forms"


def _first_drop(starts: list[float]) -> float:
    for a, b in zip(starts, starts[1:]):
        if b < a:
            return b
    return 0.0
