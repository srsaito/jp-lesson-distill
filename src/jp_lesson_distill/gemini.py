"""Thin wrapper around google-genai: file upload, structured generation, inline audio parts."""

from __future__ import annotations

import os
import time
from pathlib import Path

from google import genai
from google.genai import types


def make_client() -> genai.Client:
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set")
    return genai.Client()


def upload_audio(client: genai.Client, path: Path) -> types.File:
    f = client.files.upload(file=str(path))
    while f.state and f.state.name == "PROCESSING":
        time.sleep(3)
        f = client.files.get(name=f.name)
    if f.state and f.state.name == "FAILED":
        raise RuntimeError(f"Gemini file processing failed for {path}")
    return f


def audio_part(path: Path) -> types.Part:
    """Inline part for short clips (Files API not worth the round-trips under ~20 MB)."""
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="audio/mp4")


def generate(client: genai.Client, model: str, contents: list, schema: type, temperature: float = 0.2):
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    if resp.parsed is None:
        raise RuntimeError(f"unparsable Gemini response: {(resp.text or '<empty>')[:500]}")
    return resp.parsed
