"""Probe what the client read timeout actually measures during a thinking call (jld-hc9.7).

Runs ONE streamed structured-generation call — Pass A's prompt and schema on one audio
window — with no retry of any kind, and records every raw socket read underneath the SDK:
when it returned, how many bytes, what they began with, and which timeout httpcore passed to
that read. Alongside it, the time of every chunk the SDK actually yielded.

The question it was built for: when the stream goes quiet for longer than the read timeout
and is NOT abandoned, is the server sending bytes the SDK swallows (each resetting httpx's
per-read timer), or is the timeout not reaching the read at all?

What it found (2026-09-25): neither. No keep-alive bytes before the first chunk (the HTTP
headers arrive WITH it), and httpcore receives the configured value on every read. The
premise was a 10x misreading of the log. But it also showed that the same timeout reaches
the server as X-Server-Timeout, a deadline on the whole call — see CALL_DEADLINE_S.

    uv run python scripts/probe_read_timeout.py work/20260924/windows/w02.m4a \
        --timeout 30 --out work/probe-hc9-7/w02-t30.jsonl

Costs one Pass A window (~$0.3–0.5). Never part of the test suite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpcore._backends.sync as hc_sync
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402

from jp_lesson_distill import prompts  # noqa: E402
from jp_lesson_distill.gemini import DEFAULT_MODEL, MAX_OUTPUT_TOKENS, make_client, upload_audio  # noqa: E402
from jp_lesson_distill.models import Transcript  # noqa: E402

EVENTS: list[dict] = []
T0 = [0.0]  # set just before the generate call, so upload time is excluded


def _log(kind: str, **kw) -> None:
    t = time.monotonic() - T0[0] if T0[0] else None
    EVENTS.append({"t": None if t is None else round(t, 3), "kind": kind, **kw})


_orig_read = hc_sync.SyncStream.read


def _read(self, max_bytes, timeout=None):
    started = time.monotonic()
    try:
        data = _orig_read(self, max_bytes, timeout)
    except Exception as e:
        _log("read_error", timeout=timeout, waited=round(time.monotonic() - started, 3),
             error=type(e).__name__)
        raise
    if T0[0]:
        _log("read", timeout=timeout, waited=round(time.monotonic() - started, 3),
             nbytes=len(data), head=data[:60].decode("utf-8", "replace"))
    return data


hc_sync.SyncStream.read = _read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--timeout", type=float, default=300, help="client timeout, seconds")
    ap.add_argument("--thinking-level", default="medium")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(args.timeout)
    f = upload_audio(client, args.audio)
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=Transcript,
        temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=args.thinking_level),
    )
    T0[0] = time.monotonic()
    _log("start", timeout_s=args.timeout, model=DEFAULT_MODEL, audio=str(args.audio))
    chars = 0
    outcome = "complete"
    try:
        for chunk in client.models.generate_content_stream(
            model=DEFAULT_MODEL, contents=[f, prompts.PASS_A], config=config,
        ):
            text = chunk.text or ""
            chars += len(text)
            _log("chunk", chars=len(text), total=chars,
                 thoughts=getattr(chunk.usage_metadata, "thoughts_token_count", None))
            if len(EVENTS) % 20 == 0:
                print(f"  {EVENTS[-1]['t']:.0f}s {chars:,} chars", flush=True)
    except (httpx.TimeoutException, genai_errors.APIError) as e:
        # ReadTimeout = the client gave up on a gap; 504 DEADLINE_EXCEEDED = the server ended
        # the whole call at X-Server-Timeout (jld-hc9.7). Telling them apart is the point.
        outcome = f"{type(e).__name__}: {str(e)[:80]}"
        _log("aborted", error=outcome)
    finally:
        _log("end", outcome=outcome, chars=chars)
        with args.out.open("w") as fh:
            for ev in EVENTS:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    reads = [e for e in EVENTS if e["kind"] == "read"]
    chunks = [e for e in EVENTS if e["kind"] == "chunk"]
    print(f"{outcome}: {chars:,} chars, {len(reads)} raw reads, {len(chunks)} SDK chunks")
    if reads:
        print(f"read timeouts passed to httpcore: {sorted({r['timeout'] for r in reads})}")
        print(f"longest single read wait: {max(r['waited'] for r in reads):.1f}s")
    if chunks:
        print(f"first SDK chunk at {chunks[0]['t']:.1f}s")
    print("raw reads before the first SDK chunk:")
    first = chunks[0]["t"] if chunks else float("inf")
    for r in reads:
        if r["t"] <= first:
            print(f"  {r['t']:8.1f}s  waited {r['waited']:6.1f}s  {r['nbytes']:6d} B  {r['head']!r}")


if __name__ == "__main__":
    main()
