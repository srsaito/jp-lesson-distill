"""Thin wrapper around google-genai: file upload, structured generation, inline audio parts.

Generation is STREAMED: a full-hour Pass A response takes many minutes to generate, and a
non-streaming request sends nothing back until it's done — idle long enough that NAT/proxies
reset the connection ([Errno 54]). Streaming keeps bytes flowing the whole time.

Streaming is also the only place a degenerate decode can be caught while it is still cheap.
A repetition loop STREAMS CONTINUOUSLY — bytes keep arriving, so no timeout of any kind
fires — while it burns the whole output budget on one repeated fragment. Watching the tail
of the accumulated text is the only way to see it (jld-hc9.3).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

TRANSIENT = (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout)
RETRYABLE_CODES = {429, 500, 503}  # rate limit / transient server errors

# --- repetition-loop detection, calibrated against every transcript on disk (jld-hc9.3) ---
#
# Only one runaway loop has ever occurred: the 2026-07-29 single-pass call, which emitted
# 「持っ、」 19,452 times (58,356 garbage chars against 14,362 real ones) after transcribing
# 35 minutes healthily. The seed was legitimate — Steven starts 持っ, breaks off, restarts —
# and PASS_A asks for verbatim transcription, so emitting it once was CORRECT. Any short
# fragment can seed this; 持っ is not special.
#
# The floor has to clear real speech. Scanning 29 transcripts for the longest immediately
# repeated short unit, the worst case anywhere is 5 — and every one is a genuine Japanese
# backchannel (そうそうそうそう, はい, うん). Twenty is clear of that by a wide margin and
# still fires within ~60 characters of a three-character loop starting.
#
# The unit is short (~3 chars) and the repetition sits INSIDE a single JSON string field,
# which is exactly why quality.py's duplicate-SEGMENT warning cannot see it.
LOOP_TAIL_CHARS = 2000   # how much of the accumulated response to keep under inspection
LOOP_UNIT_MAX = 64       # longest cycle we are willing to call a loop
LOOP_MIN_REPEATS = 20    # consecutive repeats before we believe it

# --- how long a silent stream is allowed to stay silent (jld-hc9.3.2) ---
#
# A call that never sends a first chunk used to wait THIRTY MINUTES, and that was
# configuration, not a missing watchdog. The SDK turns HttpOptions(timeout=<ms>) into a
# single per-request scalar, httpx expands a scalar to all four of its fields, and httpx's
# read timeout is the maximum GAP BETWEEN CHUNKS rather than a budget for the whole
# response. So the old 30*60*1000 was, precisely, "tolerate half an hour of silence".
# Lowering the scalar is the entire fix: a long healthy stream is unaffected because the
# gap resets on every chunk, and httpx.ReadTimeout is already in TRANSIENT, so _with_retry
# re-rolls the call for free.
#
# Do NOT try per-field httpx timeouts through client_args. get_timeout_in_seconds returns
# None when http_options.timeout is unset and the SDK then passes timeout=None explicitly,
# which in httpx means "disable all timeouts" rather than "use the client default" — so
# per-field values set on the client are overridden to nothing. Verified against the
# installed SDK.
#
# PROVISIONAL VALUE. Nobody has yet measured how long a healthy audio+response_schema call
# can legitimately go before its first chunk, and thinking tokens are not answer text, so a
# window that thinks for a while looks identical to a stalled one from out here. Aborting
# too eagerly costs a whole re-transcription, so this starts deliberately generous — six
# times better than the old behaviour without pretending to a number we have not measured.
# Every call now logs first-byte latency and longest gap; jld-hc9.3.3 tightens this once
# a real lesson has supplied the distribution.
STREAM_IDLE_TIMEOUT_S = 300.0

# --- which model, and how hard it is allowed to think (jld-hc9.1) ---
#
# PINNED, not an alias. `gemini-pro-latest` is a moving target: it served Gemini 2.5 Pro when
# this pipeline was built and resolves to gemini-3.1-pro-preview today, so two runs a few days
# apart can be two different models with no sign of it in any output file. That is not
# hypothetical here — the 2026-08-18 and 2026-08-21 transcriptions of the same lesson agree on
# the words and disagree on WHO SPOKE for a whole stretch (jld-dli), and "the alias moved" was
# the leading explanation we could not rule out. A pinned id makes that question answerable:
# moments.json records exactly what transcribed the lesson.
#
# The alias is still available through --model, and _report_usage prints the version the API
# actually served whenever it differs from the id we asked for — so drift is visible either way.
DEFAULT_MODEL = "gemini-3.1-pro-preview"

# MEDIUM, and the reasoning that says `low` is measured and wrong (ADR-0007).
#
# Transcription looks mechanical — hear the words, write them down — and thinking is drawn from
# the SAME 65,536-token budget as the answer (see below), so it is tempting to turn it down.
# But Pass A does two jobs, and the second one is inference: deciding WHO SPOKE. Told to think
# less, that is the half that goes first. Three `low` runs over one real 20-minute window got
# 6 of 15 address-form cue lines wrong — short, cleanly split lines that can only be the
# student, labelled `teacher` — where the run this project measured at 89% got 5/5.
#
# `medium` is also close to the status quo: the model's own default spends ~11,500 thinking
# tokens on that window and `medium` spends ~9,300 (14% of the output budget), against a `low`
# call so cheap the API declines to report a count. The cost is about 50 s per window.
#
# Use thinking_level, never thinking_budget: on Gemini 3.x thinking_budget is silently ignored,
# and sending both is a 400. Detect and Pass B do not set a level at all — detect reasons about
# a transcript and Pass B adjudicates an error, which are the parts of this pipeline where
# thinking was never in question.
PASS_A_THINKING_LEVEL = "MEDIUM"

# What the CLI is allowed to offer. `MINIMAL` exists in the SDK enum and this model rejects it —
# "Thinking level MINIMAL is not supported for this model" (400) — so offering it would only
# hand out a flag that always fails. Re-check this list when the pin moves (ADR-0006).
THINKING_LEVELS = ("low", "medium", "high")

# Thinking tokens are billed against the OUTPUT budget, so deliberation can starve the
# response itself: a 59-minute Pass A once spent 62,911 tokens thinking and had 2,563 left
# for the transcript, truncating it ~50 segments in. That surfaced as an opaque "Invalid
# JSON: EOF while parsing", which is why the diagnostic below exists — the failure is worth
# naming, and the partial output is worth keeping (the 2026-07-29 loop sample this repo
# tests against is exactly such a dump).
MAX_OUTPUT_TOKENS = 65536

# When to complain about the thinking count. The number that matters is not "is this more than
# `low` should cost" — nobody knows what a level costs on a given minute of audio — but "is
# thinking competing with the transcript for the same budget". A quarter of the window is the
# point where it measurably is; the historical truncation was at 96%.
THINKING_WARN_SHARE = 0.25


class OutputBudgetExhausted(RuntimeError):
    """The response hit the output-token ceiling and stopped mid-JSON."""


class RepetitionLoop(RuntimeError):
    """The model started cycling on one fragment; the rest of this call is garbage."""


def find_repetition_loop(text: str) -> str | None:
    """The unit the tail is cycling on, or None.

    Checks whether the very end of `text` is periodic: the last LOOP_MIN_REPEATS
    repeats of some unit up to LOOP_UNIT_MAX long. Looking only at the tail is what
    makes this safe to call after every chunk, and it means the check keeps working
    for the whole stream — the 2026-07-29 loop began 20% in, after 92 healthy
    segments, so a warmup-only check would have missed it entirely.
    """
    tail = text[-LOOP_TAIL_CHARS:]
    for unit in range(1, LOOP_UNIT_MAX + 1):
        span = unit * LOOP_MIN_REPEATS
        if span > len(tail):
            break
        window = tail[-span:]
        if window == window[:unit] * LOOP_MIN_REPEATS:
            return window[:unit]
    return None


def _dotenv_key() -> str | None:
    """GEMINI_API_KEY from a .env in cwd or any parent (repo root when run via uv)."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        env_file = d / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"") or None
        if (d / ".git").exists():
            break
    return None


def make_client(idle_timeout_s: float = STREAM_IDLE_TIMEOUT_S) -> genai.Client:
    key = _dotenv_key()
    if key:
        print("[gemini] using GEMINI_API_KEY from .env (overrides shell env)")
    else:
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (shell env or .env in the repo)")
    print(f"[gemini] aborting any stream that goes quiet for {idle_timeout_s:.0f}s", flush=True)
    # explicit api_key so a stray GOOGLE_API_KEY in the shell can never win
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=int(idle_timeout_s * 1000)),
    )


def upload_audio(client: genai.Client, path: Path) -> types.File:
    f = _with_retry(lambda: client.files.upload(file=str(path)), "upload")
    while f.state and f.state.name == "PROCESSING":
        time.sleep(3)
        f = client.files.get(name=f.name)
    if f.state and f.state.name == "FAILED":
        raise RuntimeError(f"Gemini file processing failed for {path}")
    return f


def audio_part(path: Path) -> types.Part:
    """Inline part for short clips (Files API not worth the round-trips under ~20 MB)."""
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="audio/mp4")


def generate(client: genai.Client, model: str, contents: list, schema: type,
             temperature: float = 0.2, progress: bool = False,
             debug_dump: Path | None = None, thinking_level: str | None = None):
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if thinking_level is not None:
        # thinking_level ALONE. thinking_budget alongside it is a 400, and by itself on 3.x it
        # is accepted and ignored — which is how a Pass A call came to spend 62,911 tokens
        # thinking while its config said otherwise.
        config.thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

    def attempt():
        started = time.monotonic()
        stream = client.models.generate_content_stream(
            model=model, contents=contents, config=config,
        )
        pieces: list[str] = []
        tail = ""  # rolling window, so the check stays O(1) per chunk
        # The gaps are the measurement jld-hc9.3.3 needs: the read timeout has to sit
        # above the largest one a HEALTHY call produces, and the first is the long one.
        first_byte: float | None = None
        longest_gap = 0.0
        previous = started
        finish = None
        # Token counts arrive on whichever chunks carry usage_metadata, and a later chunk can
        # carry one that omits a field the earlier one had. Keep the last value SEEN for each
        # rather than the last usage object, or a count that was reported gets read as absent.
        thoughts: int | None = None
        answer: int | None = None
        served = None  # the model the API actually used; an alias resolves here
        for i, chunk in enumerate(stream):
            now = time.monotonic()
            longest_gap = max(longest_gap, now - previous)
            previous = now
            for candidate in chunk.candidates or []:
                if candidate.finish_reason:
                    finish = candidate.finish_reason
            if chunk.usage_metadata:
                thoughts = _latest(thoughts, chunk.usage_metadata, "thoughts_token_count")
                answer = _latest(answer, chunk.usage_metadata, "candidates_token_count")
            served = getattr(chunk, "model_version", None) or served
            if chunk.text:
                if first_byte is None:
                    first_byte = now - started
                pieces.append(chunk.text)
                tail = (tail + chunk.text)[-LOOP_TAIL_CHARS:]
                unit = find_repetition_loop(tail)
                if unit is not None:
                    if progress:
                        print(flush=True)
                    saved = _dump(debug_dump, "".join(pieces))
                    raise RepetitionLoop(
                        f"the model is cycling on {unit!r} "
                        f"({LOOP_MIN_REPEATS}+ times in a row) after "
                        f"{sum(len(p) for p in pieces):,} characters{saved}"
                    )
            if progress and i % 10 == 0:
                print(".", end="", flush=True)
        if progress:
            print(flush=True)
            text = "".join(pieces)
            print(f"[gemini] stream: {len(text):,} chars in {time.monotonic() - started:.0f}s "
                  f"(first byte {first_byte or 0:.1f}s, longest gap {longest_gap:.1f}s)",
                  flush=True)
        return "".join(pieces), finish, (thoughts, answer), served

    text, finish, (thoughts, answer), served = _with_retry(attempt, "generate")
    _report_usage(model, served, thinking_level, thoughts, answer, progress)
    if not text.strip():
        raise RuntimeError("empty Gemini response")
    if finish is not None and getattr(finish, "name", "") == "MAX_TOKENS":
        # Retrying will not help — the window asked for more output than a response can
        # hold — so say what ran out, and keep the truncated JSON to look at.
        raise OutputBudgetExhausted(
            f"the response hit the {MAX_OUTPUT_TOKENS:,}-token output ceiling and stopped "
            f"mid-JSON: {thoughts} thinking + {answer} answer tokens. Thinking is drawn from "
            "the same budget as the answer, so a shorter --window-minutes (or less thinking) "
            "is the lever, not a retry." + _dump(debug_dump, text)
        )
    return schema.model_validate_json(text)


def _latest(known: int | None, usage, field: str) -> int | None:
    seen = getattr(usage, field, None)
    return known if seen is None else seen


def _report_usage(model: str, served: str | None, thinking_level: str | None,
                  thoughts: int | None, answer: int | None, progress: bool) -> None:
    """Say what the call actually cost, and complain when thinking crowds out the answer.

    Two things are worth watching on every call, and neither is visible anywhere else:

    WHICH MODEL ANSWERED. With a pinned id this is a no-op. With an alias it is the only
    place the resolution is ever written down — `gemini-pro-latest` served 2.5 Pro when this
    was built and serves 3.1 Pro now, and nothing in a transcript records which.

    WHAT THINKING COST. thoughts_token_count is the check that `thinking_level` landed:
    thinking_budget is ignored on 3.x rather than rejected, so a config that thinks it asked
    for less can be having no effect at all, and the only evidence is the count. None means
    UNKNOWN, never zero — an absent field is the API declining to say, and treating that as
    "no thinking happened" is how you conclude a setting works when it does not.
    """
    if served and served != model:
        print(f"[gemini] served by {served} (asked for {model}) — "
              "an alias moved under you at some point; pin --model to make runs comparable",
              flush=True)
    if progress:
        asked = f", thinking_level={thinking_level.lower()}" if thinking_level else ""
        seen = "unknown" if thoughts is None else f"{thoughts:,}"
        print(f"[gemini] tokens: {seen} thinking + "
              f"{'unknown' if answer is None else f'{answer:,}'} answer "
              f"of {MAX_OUTPUT_TOKENS:,}{asked}", flush=True)
    if thinking_level is not None and thoughts is None:
        print(f"[gemini] warning: asked for thinking_level={thinking_level.lower()} and the API "
              "reported no thinking count, so whether it took effect is UNKNOWN (not zero)",
              flush=True)
    if thoughts is not None and thoughts > THINKING_WARN_SHARE * MAX_OUTPUT_TOKENS:
        print(f"[gemini] warning: {thoughts:,} thinking tokens is over "
              f"{THINKING_WARN_SHARE:.0%} of the {MAX_OUTPUT_TOKENS:,}-token output budget, "
              "which the answer shares — the transcript is being squeezed"
              + ("" if thinking_level else "; Pass A asks for thinking_level=low, "
                 "this call did not"), flush=True)


def _dump(path: Path | None, text: str) -> str:
    """Keep a failed response for inspection; the phrase to append to the error."""
    if path is None or not text:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return f". Partial output saved to {path}"


def _with_retry(fn, what: str, attempts: int = 4):
    """Retry transient failures with backoff.

    Every print here carries flush=True, which is not decoration: stdout is block-buffered
    when the run is redirected to a log, so without it the messages that explain a call still
    in flight — "the stream went quiet, retrying" — sit in a 4 KB buffer until the process
    ends. A 20-minute window that was retrying four times looked to us like one silent hang.
    """
    for n in range(1, attempts + 1):
        try:
            return fn()
        except httpx.ReadTimeout:
            # Not a network blip: the stream went quiet for longer than the client allows,
            # which is the stall this timeout exists to cut short.
            if n == attempts:
                raise
            print(f"[{what}] the stream went quiet for longer than the client allows and was "
                  f"aborted, retry {n}/{attempts - 1}…", flush=True)
            time.sleep(5 * n)
        except TRANSIENT as e:
            if n == attempts:
                raise
            print(f"[{what}] transient network error ({e.__class__.__name__}), "
                  f"retry {n}/{attempts - 1}…", flush=True)
            time.sleep(5 * n)
        except genai_errors.APIError as e:
            if getattr(e, "code", None) not in RETRYABLE_CODES or n == attempts:
                raise
            wait = 30 * n  # free-tier RPM windows are per-minute; back off generously
            print(f"[{what}] API {e.code} (rate limit/server), waiting {wait}s, "
                  f"retry {n}/{attempts - 1}…", flush=True)
            time.sleep(wait)
