"""The model pin and the thinking configuration (jld-hc9.1).

Offline; the Gemini client is faked (FakeClient comes from test_stream, which built it).

Two invariants are worth pinning against a live API that will not tell you when they break:
`thinking_budget` is ACCEPTED AND IGNORED on Gemini 3.x, so a config that sets it looks
correct and does nothing; and `gemini-pro-latest` answers happily whichever model is behind
it today. Both failure modes are silent, which is why they are tested rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jp_lesson_distill import gemini, pipeline
from jp_lesson_distill.gemini import (
    DEFAULT_MODEL,
    MAX_OUTPUT_TOKENS,
    PASS_A_THINKING_LEVEL,
    THINKING_WARN_SHARE,
    generate,
)
from jp_lesson_distill.models import Transcript
from test_stream import VALID, FakeChunk, FakeClient, FakeUsage

SRC = Path(gemini.__file__).parent


# --- the pin -------------------------------------------------------------------

def test_the_default_model_is_a_pin_not_a_moving_alias():
    """An alias is not a model: `gemini-pro-latest` served 2.5 Pro, then 3.1 Pro."""
    assert "latest" not in DEFAULT_MODEL
    assert pipeline.Config(recording=Path("x"), date="20260101").model == DEFAULT_MODEL


def test_the_model_id_is_written_down_in_exactly_one_place():
    """So a default cannot drift between the CLI, the Config and the API wrapper."""
    written = {
        path.name
        for path in SRC.glob("*.py")
        if re.search(r'"gemini-[\w.-]+"', path.read_text())
    }
    assert written == {"gemini.py"}


def test_a_model_that_answers_under_another_name_is_named(capsys):
    """The alias-drift alarm: free, and the only record of what actually answered."""
    client = FakeClient([FakeChunk(VALID)])
    client.models.chunks[0].model_version = "gemini-9.9-pro"
    generate(client, "gemini-pro-latest", ["prompt"], Transcript)
    out = capsys.readouterr().out
    assert "gemini-9.9-pro" in out and "pin --model" in out


def test_a_pinned_model_answering_as_itself_says_nothing(capsys):
    client = FakeClient([FakeChunk(VALID)])
    client.models.chunks[0].model_version = DEFAULT_MODEL
    generate(client, DEFAULT_MODEL, ["prompt"], Transcript)
    assert "asked for" not in capsys.readouterr().out


# --- thinking_level, and never thinking_budget ---------------------------------

def test_a_requested_level_reaches_the_config():
    client = FakeClient([VALID])
    generate(client, "fake-model", ["prompt"], Transcript, thinking_level="LOW")
    thinking = client.models.last_kwargs["config"].thinking_config
    assert thinking.thinking_level == "LOW"


def test_thinking_budget_is_never_sent_alongside_a_level():
    """Sending both is a 400, and on 3.x the budget is the half that does nothing."""
    client = FakeClient([VALID])
    generate(client, "fake-model", ["prompt"], Transcript, thinking_level="LOW")
    assert client.models.last_kwargs["config"].thinking_config.thinking_budget is None


def test_thinking_budget_is_never_set_anywhere_in_the_source():
    """It is a no-op on 3.x; re-introducing it would quietly do nothing (bd memory).

    The prose in gemini.py names it — that is the point of the prose — so this looks for
    the keyword argument, which is the only form that would reach the API.
    """
    assert not [p.name for p in SRC.glob("*.py") if "thinking_budget=" in p.read_text()]


def test_no_level_means_no_thinking_config_at_all():
    """Detect and Pass B reason for a living; they keep the model's own default."""
    client = FakeClient([VALID])
    generate(client, "fake-model", ["prompt"], Transcript)
    assert client.models.last_kwargs["config"].thinking_config is None


def test_pass_a_sends_the_configured_level(monkeypatch, tmp_path):
    """The wiring, not the constant: the level has to reach the transcription call."""
    seen = {}

    def fake_generate(client, model, contents, schema, **kwargs):
        seen.update(kwargs)
        return Transcript.model_validate_json(VALID)

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    monkeypatch.setattr(pipeline, "upload_audio", lambda client, path: "uploaded")
    cfg = pipeline.Config(recording=tmp_path / "r.m4a", date="20260101")
    pipeline._transcribe_window(cfg, lambda: FakeClient([]), tmp_path / "w.m4a",
                                "window 1/1", "00:00-00:30", 30.0, tmp_path / "t.json")
    assert seen["thinking_level"] == PASS_A_THINKING_LEVEL


def test_pass_a_does_not_default_to_low():
    """Measured, not assumed (ADR-0007). Three `low` runs over one real 20-minute window got
    6 of 15 address-form cue lines wrong — clean, well-split lines that can only be the
    student, labelled teacher — against 5/5 for the run that scored 89% on blind ground truth.
    Transcription is mechanical; deciding who spoke is not, and that is the half `low` spends."""
    assert PASS_A_THINKING_LEVEL != "LOW"


def test_the_level_can_be_turned_off_for_a_comparison(monkeypatch, tmp_path):
    """--pass-a-thinking model-default; how the LOW-vs-default measurement was taken."""
    seen = {}

    def fake_generate(client, model, contents, schema, **kwargs):
        seen.update(kwargs)
        return Transcript.model_validate_json(VALID)

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    monkeypatch.setattr(pipeline, "upload_audio", lambda client, path: "uploaded")
    cfg = pipeline.Config(recording=tmp_path / "r.m4a", date="20260101", pass_a_thinking=None)
    pipeline._transcribe_window(cfg, lambda: FakeClient([]), tmp_path / "w.m4a",
                                "window 1/1", "00:00-00:30", 30.0, tmp_path / "t.json")
    assert seen["thinking_level"] is None


# --- the instrumentation -------------------------------------------------------

def _run_with_usage(thoughts, answer=4_000, level="LOW", progress=False):
    client = FakeClient([FakeChunk(VALID, usage=FakeUsage(thoughts=thoughts, answer=answer))])
    generate(client, "fake-model", ["prompt"], Transcript,
             thinking_level=level, progress=progress)


def test_the_token_split_is_logged_so_a_level_can_be_checked(capsys):
    _run_with_usage(1_200, progress=True)
    out = capsys.readouterr().out
    assert "1,200 thinking" in out and "thinking_level=low" in out


def test_an_absent_thinking_count_is_unknown_not_zero(capsys):
    """The trap: reading a missing field as 0 'confirms' a setting that did nothing."""
    _run_with_usage(None)
    out = capsys.readouterr().out
    assert "UNKNOWN" in out and "not zero" in out


def test_a_count_reported_early_survives_a_later_chunk_that_omits_it(capsys):
    """Usage arrives chunk by chunk; the last chunk is not necessarily the fullest.

    Reading only the final usage_metadata reported "unknown thinking" on a real 20-minute
    window that had in fact reported a count, which would have made the warning below
    fire on healthy calls and taught us to ignore it.
    """
    client = FakeClient([
        FakeChunk(VALID[:20], usage=FakeUsage(thoughts=1_100, answer=None)),
        FakeChunk(VALID[20:], usage=FakeUsage(thoughts=None, answer=8_500)),
    ])
    generate(client, "fake-model", ["prompt"], Transcript, thinking_level="LOW", progress=True)
    out = capsys.readouterr().out
    assert "1,100 thinking + 8,500 answer" in out
    assert "UNKNOWN" not in out


def test_thinking_that_crowds_out_the_transcript_warns(capsys):
    _run_with_usage(int(THINKING_WARN_SHARE * MAX_OUTPUT_TOKENS) + 1)
    assert "output budget" in capsys.readouterr().out


def test_a_modest_thinking_count_is_silent(capsys):
    _run_with_usage(1_200)
    assert "warning" not in capsys.readouterr().out


def test_the_historical_truncation_would_have_warned_long_before_it_truncated(capsys):
    """62,911 thinking tokens left 2,563 for an hour of transcript (jld-hc9.3.4)."""
    _run_with_usage(62_911, answer=2_563, level=None)
    out = capsys.readouterr().out
    assert "62,911" in out and "Pass A asks for thinking_level=low" in out


def test_every_level_the_cli_offers_is_one_the_sdk_knows():
    from google.genai import types

    known = {member.value for member in types.ThinkingLevel}
    assert {level.upper() for level in gemini.THINKING_LEVELS} <= known
    assert PASS_A_THINKING_LEVEL.lower() in gemini.THINKING_LEVELS


def test_the_level_this_model_rejects_is_not_on_the_menu():
    """MINIMAL is in the SDK enum and gemini-3.1-pro-preview answers it with a 400:
    "Thinking level MINIMAL is not supported for this model". Offering it would be a flag
    that never works. Re-check when the pin moves."""
    assert "minimal" not in gemini.THINKING_LEVELS
