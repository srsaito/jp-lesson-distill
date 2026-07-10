"""Thin wrapper around google-genai: file upload, structured generation, inline audio parts.

Generation is STREAMED: a full-hour Pass A response takes many minutes to generate, and a
non-streaming request sends nothing back until it's done — idle long enough that NAT/proxies
reset the connection ([Errno 54]). Streaming keeps bytes flowing the whole time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import types

TRANSIENT = (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout)


def make_client() -> genai.Client:
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set")
    return genai.Client(http_options=types.HttpOptions(timeout=30 * 60 * 1000))


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
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        pieces = []
        for i, chunk in enumerate(stream):
            if chunk.text:
                pieces.append(chunk.text)
            if progress and i % 10 == 0:
                print(".", end="", flush=True)
        if progress:
            print(flush=True)
        return "".join(pieces)

    text = _with_retry(attempt, "generate")
    if not text.strip():
        raise RuntimeError("empty Gemini response")
    return schema.model_validate_json(text)


def _with_retry(fn, what: str, attempts: int = 3):
    for n in range(1, attempts + 1):
        try:
            return fn()
        except TRANSIENT as e:
            if n == attempts:
                raise
            print(f"[{what}] transient network error ({e.__class__.__name__}), retry {n}/{attempts - 1}…")
            time.sleep(5 * n)
