"""Streaming behaviour of gemini.generate: the idle timeout and the loop abort.

Offline. The Gemini client is faked — these tests are about what OUR code does with
a stream, not about the API.

Several tests here pin behaviour of httpx and google-genai that the fix depends on.
That is deliberate: the reason a stalled call used to hang for thirty minutes is that
a scalar timeout silently becomes a per-gap READ timeout, and the reason the obvious
per-field alternative does not work is that the SDK passes an explicit None. If either
ever changes, the timeout fix stops meaning what its comment says it means.
"""

from __future__ import annotations

import httpx
import pytest
from google.genai import _api_client as genai_api
from google.genai import types

from jp_lesson_distill import gemini
from jp_lesson_distill.gemini import (
    LOOP_MIN_REPEATS,
    STREAM_IDLE_TIMEOUT_S,
    RepetitionLoop,
    generate,
    make_client,
)
from jp_lesson_distill.models import Transcript


class FakeChunk:
    def __init__(self, text: str, finish_reason=None, usage=None):
        self.text = text
        self.candidates = [FakeCandidate(finish_reason)] if finish_reason else None
        self.usage_metadata = usage


class FakeCandidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class FakeReason:
    def __init__(self, name: str):
        self.name = name


class FakeUsage:
    def __init__(self, thoughts: int, answer: int):
        self.thoughts_token_count = thoughts
        self.candidates_token_count = answer


class FakeModels:
    """Replays a scripted stream; a chunk may be an exception to raise instead."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def generate_content_stream(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs

        def stream():
            for c in self.chunks:
                if isinstance(c, Exception):
                    raise c
                yield c if isinstance(c, FakeChunk) else FakeChunk(c)

        return stream()


class FakeClient:
    def __init__(self, chunks):
        self.models = FakeModels(chunks)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """_with_retry sleeps between attempts; the wait is not what is under test."""
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)


# --- the assumptions the timeout fix rests on ---------------------------------

def test_a_scalar_timeout_becomes_a_per_gap_read_timeout():
    """Why 30*60*1000 meant 'tolerate half an hour of silence'."""
    request = httpx.Client().build_request("GET", "https://example.invalid", timeout=1800)
    assert request.extensions["timeout"] == {
        "connect": 1800, "read": 1800, "write": 1800, "pool": 1800,
    }


def test_an_explicit_none_disables_every_timeout():
    """The trap that makes per-field httpx timeouts useless through this SDK.

    get_timeout_in_seconds returns None when http_options.timeout is unset, and the
    SDK then passes timeout=None explicitly — which httpx reads as 'no timeouts at
    all', not 'use the client default'.
    """
    request = httpx.Client().build_request("GET", "https://example.invalid", timeout=None)
    assert set(request.extensions["timeout"].values()) == {None}
    assert genai_api.get_timeout_in_seconds(types.HttpOptions().timeout) is None


def test_the_sdk_reads_http_options_timeout_as_milliseconds():
    assert genai_api.get_timeout_in_seconds(types.HttpOptions(timeout=120_000).timeout) == 120.0


def test_client_carries_the_configured_idle_timeout(monkeypatch):
    monkeypatch.setattr(gemini, "_dotenv_key", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-one")
    client = make_client(120.0)
    assert genai_api.get_timeout_in_seconds(client._api_client._http_options.timeout) == 120.0


def test_the_default_is_far_below_the_thirty_minutes_it_replaced():
    assert STREAM_IDLE_TIMEOUT_S < 30 * 60


# --- what generate() does with a stream ---------------------------------------

VALID = '{"segments": [{"start": "00:01", "speaker": "teacher", "text": "はい。"}]}'


def test_a_clean_stream_parses():
    client = FakeClient([VALID[:20], VALID[20:]])
    out = generate(client, "fake-model", ["prompt"], Transcript)
    assert out.segments[0].text == "はい。"


def test_a_loop_aborts_the_call_without_finishing_the_stream():
    """The abort must not wait for the model to stop on its own — that is the saving."""
    tail = ["持っ、"] * (LOOP_MIN_REPEATS * 50)
    client = FakeClient(['{"segments": [{"start": "00:01", "text": "'] + tail)
    with pytest.raises(RepetitionLoop) as caught:
        generate(client, "fake-model", ["prompt"], Transcript)
    assert "持っ、" in str(caught.value)


def test_a_loop_is_not_retried_inside_generate():
    """pipeline._transcribe_window owns the retry; a second one here would double it."""
    client = FakeClient(['{"segments": [{"text": "'] + ["ん"] * (LOOP_MIN_REPEATS * 3))
    with pytest.raises(RepetitionLoop):
        generate(client, "fake-model", ["prompt"], Transcript)
    assert client.models.calls == 1


def test_a_stalled_stream_is_retried_and_then_gives_up(capsys):
    """A read timeout is the stall surfacing; _with_retry re-rolls the whole call."""
    client = FakeClient([httpx.ReadTimeout("timed out")])
    with pytest.raises(httpx.ReadTimeout):
        generate(client, "fake-model", ["prompt"], Transcript)
    assert client.models.calls > 1
    assert "went quiet" in capsys.readouterr().out


def test_a_stall_that_clears_on_the_retry_succeeds():
    client = FakeClient([httpx.ReadTimeout("timed out")])

    def second_time_lucky(**kwargs):
        client.models.calls += 1
        if client.models.calls == 1:
            raise httpx.ReadTimeout("timed out")
        return iter([FakeChunk(VALID)])

    client.models.generate_content_stream = second_time_lucky
    out = generate(client, "fake-model", ["prompt"], Transcript)
    assert out.segments[0].text == "はい。"


# --- running out of output budget (jld-hc9.3.4) --------------------------------

TRUNCATED = '{"segments": [{"start": "00:01", "speaker": "teacher", "text": "はい'


def test_the_output_ceiling_is_asked_for_explicitly():
    client = FakeClient([VALID])
    generate(client, "fake-model", ["prompt"], Transcript)
    assert client.models.last_kwargs["config"].max_output_tokens == gemini.MAX_OUTPUT_TOKENS


def test_a_truncated_response_says_what_ran_out():
    """Without this it surfaced as 'Invalid JSON: EOF while parsing'."""
    client = FakeClient([
        FakeChunk(TRUNCATED, finish_reason=FakeReason("MAX_TOKENS"),
                  usage=FakeUsage(thoughts=62_911, answer=2_563)),
    ])
    with pytest.raises(gemini.OutputBudgetExhausted) as caught:
        generate(client, "fake-model", ["prompt"], Transcript)
    message = str(caught.value)
    assert "62911 thinking" in message and "2563 answer" in message
    assert "window-minutes" in message  # the lever that actually helps


def test_a_truncated_response_is_not_retried():
    """A window too big to fit in one response will not fit on the second try either."""
    client = FakeClient([
        FakeChunk(TRUNCATED, finish_reason=FakeReason("MAX_TOKENS"),
                  usage=FakeUsage(thoughts=1, answer=1)),
    ])
    with pytest.raises(gemini.OutputBudgetExhausted):
        generate(client, "fake-model", ["prompt"], Transcript)
    assert client.models.calls == 1


def test_a_normal_finish_reason_is_left_alone():
    client = FakeClient([FakeChunk(VALID, finish_reason=FakeReason("STOP"))])
    assert generate(client, "fake-model", ["prompt"], Transcript).segments


def test_the_partial_is_kept_for_inspection(tmp_path):
    dump = tmp_path / "nested" / "pass_a_partial.json"
    client = FakeClient([
        FakeChunk(TRUNCATED, finish_reason=FakeReason("MAX_TOKENS"),
                  usage=FakeUsage(thoughts=1, answer=1)),
    ])
    with pytest.raises(gemini.OutputBudgetExhausted) as caught:
        generate(client, "fake-model", ["prompt"], Transcript, debug_dump=dump)
    assert dump.read_text() == TRUNCATED
    assert str(dump) in str(caught.value)


def test_a_loop_keeps_its_evidence_too(tmp_path):
    """This is how work/20260729/pass_a_partial.attempt1.json came to exist."""
    dump = tmp_path / "pass_a_partial.json"
    client = FakeClient(['{"segments": [{"text": "'] + ["持っ、"] * (LOOP_MIN_REPEATS * 2))
    with pytest.raises(RepetitionLoop):
        generate(client, "fake-model", ["prompt"], Transcript, debug_dump=dump)
    assert "持っ、" * LOOP_MIN_REPEATS in dump.read_text()
