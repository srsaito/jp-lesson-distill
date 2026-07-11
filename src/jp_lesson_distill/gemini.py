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
from google.genai import errors as genai_errors
from google.genai import types

TRANSIENT = (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout)
RETRYABLE_CODES = {429, 500, 503}  # rate limit / transient server errors


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


def make_client() -> genai.Client:
    key = _dotenv_key()
    if key:
        print("[gemini] using GEMINI_API_KEY from .env (overrides shell env)")
    else:
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (shell env or .env in the repo)")
    # explicit api_key so a stray GOOGLE_API_KEY in the shell can never win
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30 * 60 * 1000))


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


def _with_retry(fn, what: str, attempts: int = 4):
    for n in range(1, attempts + 1):
        try:
            return fn()
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
