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
    print(f"[gemini] aborting any stream that goes quiet for {idle_timeout_s:.0f}s")
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
             temperature: float = 0.2, progress: bool = False):
    def attempt():
        started = time.monotonic()
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        pieces: list[str] = []
        tail = ""  # rolling window, so the check stays O(1) per chunk
        # The gaps are the measurement jld-hc9.3.3 needs: the read timeout has to sit
        # above the largest one a HEALTHY call produces, and the first is the long one.
        first_byte: float | None = None
        longest_gap = 0.0
        previous = started
        for i, chunk in enumerate(stream):
            now = time.monotonic()
            longest_gap = max(longest_gap, now - previous)
            previous = now
            if chunk.text:
                if first_byte is None:
                    first_byte = now - started
                pieces.append(chunk.text)
                tail = (tail + chunk.text)[-LOOP_TAIL_CHARS:]
                unit = find_repetition_loop(tail)
                if unit is not None:
                    if progress:
                        print(flush=True)
                    raise RepetitionLoop(
                        f"the model is cycling on {unit!r} "
                        f"({LOOP_MIN_REPEATS}+ times in a row) after "
                        f"{sum(len(p) for p in pieces):,} characters"
                    )
            if progress and i % 10 == 0:
                print(".", end="", flush=True)
        if progress:
            print(flush=True)
            text = "".join(pieces)
            print(f"[gemini] stream: {len(text):,} chars in {time.monotonic() - started:.0f}s "
                  f"(first byte {first_byte or 0:.1f}s, longest gap {longest_gap:.1f}s)")
        return "".join(pieces)

    text = _with_retry(attempt, "generate")
    if not text.strip():
        raise RuntimeError("empty Gemini response")
    return schema.model_validate_json(text)


def _with_retry(fn, what: str, attempts: int = 4):
    for n in range(1, attempts + 1):
        try:
            return fn()
        except httpx.ReadTimeout:
            # Not a network blip: the stream went quiet for longer than the client allows,
            # which is the stall this timeout exists to cut short.
            if n == attempts:
                raise
            print(f"[{what}] the stream went quiet for longer than the client allows and was "
                  f"aborted, retry {n}/{attempts - 1}…")
            time.sleep(5 * n)
        except TRANSIENT as e:
            if n == attempts:
                raise
            print(f"[{what}] transient network error ({e.__class__.__name__}), retry {n}/{attempts - 1}…")
            time.sleep(5 * n)
        except genai_errors.APIError as e:
            if getattr(e, "code", None) not in RETRYABLE_CODES or n == attempts:
                raise
            wait = 30 * n  # free-tier RPM windows are per-minute; back off generously
            print(f"[{what}] API {e.code} (rate limit/server), waiting {wait}s, retry {n}/{attempts - 1}…")
            time.sleep(wait)
