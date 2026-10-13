"""Default models and the calls each adapter makes with them.

The defaults are the cheapest current models that are good enough for a
conversational agent, pinned by exact id. These tests pin the request shapes the
current vendor APIs expect for them.
"""

import base64
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.providers.gemini_provider import GeminiLLMProvider, GeminiSTTProvider, GeminiTTSProvider
from src.providers.openai_provider import OpenAILLMProvider


def _wav(n_bytes: int) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", n_bytes) + b"\x00" * n_bytes)


@pytest.fixture
def gemini(mocker):
    client = MagicMock()
    client.aio.interactions.create = AsyncMock()
    mocker.patch("src.providers.gemini_provider.GeminiClientSingleton.get_client",
                 return_value=client)
    return client


def test_gemini_defaults_are_pinned(monkeypatch):
    for var in ("LLM", "STT", "TTS"):
        monkeypatch.delenv(f"OVARP_GEMINI_{var}_MODEL", raising=False)
    assert GeminiLLMProvider().model == "gemini-3.5-flash-lite"
    assert GeminiSTTProvider().model == "gemini-3.5-flash-lite"
    tts = GeminiTTSProvider()
    assert tts.model == "gemini-3.8-flash-lite-tts"
    assert tts.uses_interactions


@pytest.mark.asyncio
async def test_gemini_tts_uses_interactions_and_speaks_the_text_as_given(gemini):
    audio = _wav(9600)
    gemini.aio.interactions.create.return_value = SimpleNamespace(
        output_audio=SimpleNamespace(data=base64.b64encode(audio).decode(), mime_type="audio/wav"))
    tts = GeminiTTSProvider(model_name="gemini-3.8-flash-lite-tts", voice="Puck")

    chunks = [c async for c in tts.synthesize_stream("Hola, ¿cómo estás?")]

    assert b"".join(chunks) == audio  # already a WAV: no second header
    kwargs = gemini.aio.interactions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash-lite-tts"
    assert kwargs["input"] == "Hola, ¿cómo estás?"  # no prompt wrapper to be read aloud
    assert kwargs["response_format"] == {"type": "audio"}
    assert kwargs["generation_config"] == {"speech_config": [{"voice": "Puck"}]}


@pytest.mark.asyncio
async def test_gemini_tts_adds_a_header_to_raw_pcm(gemini):
    pcm = b"\x01\x00" * 2400
    gemini.aio.interactions.create.return_value = SimpleNamespace(
        output_audio=SimpleNamespace(data=pcm, mime_type="audio/l16"))
    chunks = [c async for c in GeminiTTSProvider().synthesize_stream("Hi.")]
    assert chunks[0].startswith(b"RIFF") and b"".join(chunks[1:]) == pcm


@pytest.mark.asyncio
async def test_gemini_tts_failure_is_raised_not_swallowed(gemini):
    # The orchestrator turns this into a pipeline_error on the console.
    gemini.aio.interactions.create.side_effect = RuntimeError("quota exceeded")
    with pytest.raises(RuntimeError):
        async for _ in GeminiTTSProvider().synthesize_stream("Hi."):
            pass


@pytest.mark.asyncio
async def test_gemini_tts_empty_audio_is_an_error(gemini):
    gemini.aio.interactions.create.return_value = SimpleNamespace(output_audio=None)
    with pytest.raises(RuntimeError):
        async for _ in GeminiTTSProvider().synthesize_stream("Hi."):
            pass


@pytest.mark.asyncio
async def test_gemini_llm_sets_no_temperature(mocker, mock_config):
    client = MagicMock()
    reply = MagicMock(candidates=[], text="")
    client.models.generate_content = MagicMock(return_value=reply)
    mocker.patch("src.providers.gemini_provider.GeminiClientSingleton.get_client",
                 return_value=client)
    llm = GeminiLLMProvider()
    await llm.generate_response_with_actions("hi", system_prompt="Be brief.")
    await llm.extract_actions("Hello there.", system_prompt="Be brief.")
    for call in client.models.generate_content.call_args_list:
        assert call.kwargs["config"].temperature is None


@pytest.fixture
def openai_client(mocker):
    client = MagicMock()
    msg = MagicMock(content="Hi.", tool_calls=[])
    response = MagicMock(choices=[MagicMock(message=msg)])
    client.chat.completions.create = AsyncMock(return_value=response)
    mocker.patch("src.providers.openai_provider.OpenAIClientSingleton.get_client",
                 return_value=client)
    return client


@pytest.mark.asyncio
async def test_openai_calls_ask_for_no_reasoning_by_default(openai_client, monkeypatch,
                                                           mock_config):
    monkeypatch.delenv("OVARP_OPENAI_REASONING_EFFORT", raising=False)
    llm = OpenAILLMProvider()
    await llm.generate_response_with_actions("hi")
    await llm.extract_actions("Hello there.")
    for call in openai_client.chat.completions.create.call_args_list:
        assert call.kwargs["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_reasoning_effort_can_be_left_out_for_older_models(openai_client, monkeypatch,
                                                                 mock_config):
    monkeypatch.setenv("OVARP_OPENAI_REASONING_EFFORT", "")
    await OpenAILLMProvider(model_name="gpt-4o").generate_response_with_actions("hi")
    assert "reasoning_effort" not in openai_client.chat.completions.create.call_args.kwargs


def test_console_is_told_the_model_each_provider_calls(monkeypatch):
    from fastapi.testclient import TestClient

    import src.main as main_module
    from src.core.runtime import runtime

    orch = MagicMock()
    orch.llm_providers = {"gemini": GeminiLLMProvider(model_name="gemini-3.5-flash-lite"),
                          "openai": OpenAILLMProvider(model_name="gpt-6-luna")}
    orch.tts_providers, orch.stt_providers = {}, {}
    orch.active_llm_id = "gemini"
    orch.system_prompt = ""
    monkeypatch.setattr(runtime, "orchestrator", orch, raising=False)
    monkeypatch.setattr("src.api.routers.llm._agents_overriding_global", lambda: [])
    data = TestClient(main_module.app).get("/api/llm/config").json()
    assert data["llm_models"] == {"gemini": "gemini-3.5-flash-lite", "openai": "gpt-6-luna"}
