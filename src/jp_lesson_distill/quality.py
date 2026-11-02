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

# For messages only — derived so the text can never list a different set than the check uses.
MARKER_LIST = "".join(f"「{m}」" for m, _ in MARKERS)
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
            f"only {agree}/{n_cued} segments containing {MARKER_LIST} "
            f"are labelled the way the address form requires — diarization "
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
        return (f"no {MARKER_LIST} lines in this lesson, so "
                "diarization is unverified — normal for a drill-heavy hour")
    head = f"address forms {agree}/{total}"
    if total < MARKER_MIN_N:
        return f"{head} — too few cues to judge diarization either way"
    if agree / total < MARKER_WARN:
        return (f"{head} — diarization looks unreliable, and the damage is not limited "
                f"to these {total} lines. Cut a listening kit: `distill label <recording> "
                "--date <date>` then `--play`, and `distill score` to measure it")
    return f"{head} — diarization consistent with the address forms"


def marker_by_window(segments: list[Segment], bounds: list[tuple[float, float]]) -> str:
    """The address-form check per Pass A window (jld-ec6, part 1).

    Each window is diarized by its own call, so an inversion is a per-window property, and the
    pooled verdict above can read "consistent" while a window with no cues is inverted (10/14:
    9/11 overall, window 3 almost entirely swapped). This line names the windows nothing checked.
    """
    parts, unverified, doubtful = [], [], []
    for w, (lo, hi) in enumerate(bounds, start=1):
        inside = [s for s in segments if lo <= parse_ts(s.start) < hi]
        agree, total = marker_agreement(inside)
        parts.append(f"w{w:02d} {agree}/{total}" if total else f"w{w:02d} none")
        if not total:
            unverified.append(w)
        elif total >= MARKER_MIN_N and agree / total < MARKER_WARN:
            doubtful.append(w)
    line = "address forms by window: " + " · ".join(parts)
    if doubtful:
        line += (f" — window{'s' if len(doubtful) > 1 else ''} {', '.join(map(str, doubtful))} "
                 f"look{'' if len(doubtful) > 1 else 's'} inverted or mixed")
    if unverified:
        line += (f" — window{'s' if len(unverified) > 1 else ''} "
                 f"{', '.join(map(str, unverified))} unverified (no cues)")
    return line


# --- per-track windows (ADR-0010, jld-8nv.3) ---
#
# evaluate() cannot judge a single speaker's track: it is silent much of the time by design, so
# coverage and dead time mean nothing, and segment counts vary run to run with nothing wrong
# (teacher 10/26 w3: 81 and 192 segments, same text). The track itself says how much speech there
# was, so the transcript is judged against that. Calibrated on all 38 per-track windows on disk
# (10/12, 10/14, 10/25, 10/26; counts in tests/fixtures/track_gates_calibration.json):
#   chars per second of speech  3.66-5.83, whichever lesson or speaker
#   speech behind one start     at most 76 s (long segments are fine, holes are not)
#   lines on silence            at most 6.6%, except the one defective window, at 22.6%
TRACK_MIN_SPEECH_S = 30.0    # below this a rate says nothing (a short tail, a quiet window)
TRACK_RATE_FLOOR = 2.5       # a silent compression returns about a third: ~1.4 chars/s
TRACK_RATE_CEILING = 9.0     # text well beyond what was said is invented or repeated
TRACK_MAX_SPEECH_GAP_S = 150.0
TRACK_GHOST_FRAC = 0.15
TRACK_GHOST_MIN_N = 5        # one stray うん in a four-line window is not a timing failure
GHOST_BEFORE_S, GHOST_AFTER_S = 2.0, 3.0  # how far from a start its voice may be


def is_ghost(db, frame_s: float, sound_db: float, t: float) -> bool:
    """No sound on the line's own track near the model's ORIGINAL start.

    Use it before snapping: snapping moves an invented line onto whatever sound is nearby.
    """
    near = db[max(0, int((t - GHOST_BEFORE_S) / frame_s)): int((t + GHOST_AFTER_S) / frame_s)]
    return len(near) > 0 and bool(near.max() < sound_db)


@dataclass
class TrackReport:
    speech_s: float
    segments: int
    chars: int
    max_speech_gap: float
    ghosts: int
    checks: list[Check] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.chars / self.speech_s if self.speech_s else 0.0

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.failed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        return (f"{self.segments} segments, {self.speech_s:.0f}s of speech, {self.rate:.1f} chars/s, "
                f"up to {self.max_speech_gap:.0f}s of speech behind one start, "
                f"{self.ghosts} on silence")


def evaluate_track(starts: list[float], texts: list[str], db, frame_s: float, sound_db: float,
                   offset: float, duration: float) -> TrackReport:
    """Judge one speaker's window against how much that speaker actually spoke.

    `starts` are window-relative and as the model gave them; `db` is the whole track's energy.
    """
    lo, hi = int(offset / frame_s), min(len(db), int((offset + duration) / frame_s))
    active = db[lo:hi] > sound_db
    speech = float(active.sum()) * frame_s
    chars = sum(len(t) for t in texts)
    cuts = sorted({min(len(active), max(0, int(t / frame_s))) for t in starts})
    edges = [0, *cuts, len(active)]
    gap = max((float(active[a:b].sum()) * frame_s for a, b in zip(edges, edges[1:])), default=0.0)
    ghosts = sum(is_ghost(db, frame_s, sound_db, offset + t) for t in starts)
    report = TrackReport(speech, len(texts), chars, gap, ghosts)
    enough = speech >= TRACK_MIN_SPEECH_S
    report.checks = [
        Check("text-missing", not enough or report.rate >= TRACK_RATE_FLOOR,
              f"{report.rate:.1f} chars per second of speech, under {TRACK_RATE_FLOOR:g}: "
              "text is missing"),
        Check("text-invented", not enough or report.rate <= TRACK_RATE_CEILING,
              f"{report.rate:.1f} chars per second of speech, over {TRACK_RATE_CEILING:g}: "
              "more text than was said"),
        Check("dropout", gap <= TRACK_MAX_SPEECH_GAP_S,
              f"{gap:.0f}s of speech behind one start, over {TRACK_MAX_SPEECH_GAP_S:g}s"),
        Check("timing", len(starts) < TRACK_GHOST_MIN_N or ghosts / len(starts) <= TRACK_GHOST_FRAC,
              f"{ghosts} of {len(starts)} lines start where this track is silent: "
              "the timestamps have drifted or the lines are invented"),
    ]
    return report


def _first_drop(starts: list[float]) -> float:
    for a, b in zip(starts, starts[1:]):
        if b < a:
            return b
    return 0.0
