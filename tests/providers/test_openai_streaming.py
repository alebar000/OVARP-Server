"""Streaming capability of the OpenAI-compatible adapters.

With ``stream_reply`` and ``extract_actions`` the orchestrator speaks sentence by
sentence while the reply is still being generated, and classifies gestures and
emotion off the audio path. These tests pin the calls each adapter makes.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.orchestrator import DialogOrchestrator
from src.providers.custom_provider import CustomLLMProvider
from src.providers.openai_provider import (
    NAMED_TOOL_CHOICE,
    PLAIN_TEXT_OVERRIDE,
    OpenAILLMProvider,
    actions_only_tools,
)


def _chunk(text):
    """A streamed chat-completion chunk; ``None`` makes a usage-only chunk."""
    if text is None:
        return SimpleNamespace(choices=[])
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _Stream:
    def __init__(self, items, fail_after=None):
        self._items = list(items)
        self._fail_after = fail_after

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, item in enumerate(self._items):
            if self._fail_after is not None and i == self._fail_after:
                raise RuntimeError("connection reset")
            yield item


def _tool_response(args):
    call = MagicMock()
    call.function.name = "update_agent_state"
    call.function.arguments = json.dumps(args)
    msg = MagicMock(content=None, tool_calls=[call])
    return MagicMock(choices=[MagicMock(message=msg)])


@pytest.fixture
def client(mocker):
    c = MagicMock()
    c.chat.completions.create = AsyncMock()
    mocker.patch("src.providers.openai_provider.OpenAIClientSingleton.get_client", return_value=c)
    return c


def test_adapters_offer_the_streaming_capability():
    # The orchestrator probes with hasattr, so this is what selects the sentence path.
    for llm in (OpenAILLMProvider(), CustomLLMProvider("local", "http://localhost:1/v1", "m")):
        assert hasattr(llm, "stream_reply") and hasattr(llm, "extract_actions")


@pytest.mark.asyncio
async def test_stream_reply_yields_text_as_it_arrives(client):
    client.chat.completions.create.return_value = _Stream(
        [_chunk("Hello"), _chunk(""), _chunk(" there."), _chunk(None)])
    out = [d async for d in OpenAILLMProvider().stream_reply(
        "hi", system_prompt="Always call update_agent_state.",
        history=[{"role": "assistant", "content": "earlier"}])]
    assert out == ["Hello", " there."]

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert "tools" not in kwargs  # a forced tool call would hold the text back
    assert kwargs["messages"][0]["content"].endswith(PLAIN_TEXT_OVERRIDE)
    assert kwargs["messages"][1] == {"role": "assistant", "content": "earlier"}
    assert kwargs["messages"][-1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_stream_failure_propagates(client):
    client.chat.completions.create.return_value = _Stream(
        [_chunk("Hel"), _chunk("lo")], fail_after=1)
    with pytest.raises(RuntimeError):
        async for _ in OpenAILLMProvider().stream_reply("hi"):
            pass


@pytest.mark.asyncio
async def test_extract_actions_classifies_without_rewriting(client):
    client.chat.completions.create.return_value = _tool_response(
        {"emotions": "happy", "actions": "wave", "spoken_response": "should be dropped"})
    actions = await OpenAILLMProvider().extract_actions("Nice to meet you.", "Be warm.")
    assert actions == {"emotions": "happy", "actions": "wave"}

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == NAMED_TOOL_CHOICE
    params = kwargs["tools"][0]["function"]["parameters"]
    assert "spoken_response" not in params["properties"]
    assert "Nice to meet you." in kwargs["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_extract_actions_failure_yields_no_actions(client):
    client.chat.completions.create.side_effect = RuntimeError("rate limited")
    assert await OpenAILLMProvider().extract_actions("Hi.") == {}


def test_single_value_category_is_offered_not_required(mock_config):
    mock_config.custom_commands["avatar"] = type(mock_config.custom_commands["emotions"])(
        description="desc", values=["default"])
    params = actions_only_tools()[0]["function"]["parameters"]
    assert "avatar" in params["properties"]
    assert "avatar" not in params["required"]
    assert "emotions" in params["required"]


@pytest.mark.asyncio
async def test_custom_endpoint_falls_back_to_required_tool_choice():
    llm = CustomLLMProvider("local", "http://localhost:1/v1", "m")
    llm.client = MagicMock()
    llm.client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("named tool_choice unsupported"),
                     _tool_response({"emotions": "sad"})])
    assert await llm.extract_actions("Oh no.") == {"emotions": "sad"}
    choices = [c.kwargs["tool_choice"] for c in llm.client.chat.completions.create.call_args_list]
    assert choices == [NAMED_TOOL_CHOICE, "required"]


@pytest.mark.asyncio
async def test_custom_endpoint_without_tools_yields_no_actions():
    llm = CustomLLMProvider("local", "http://localhost:1/v1", "m")
    llm.client = MagicMock()
    llm.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("no tools"))
    assert await llm.extract_actions("Hi.") == {}


@pytest.mark.asyncio
async def test_openai_reply_is_spoken_sentence_by_sentence(client, mocker):
    """End to end through the orchestrator: text chunks go out as they arrive and
    each sentence is synthesized on its own, before the reply has finished."""
    client.chat.completions.create.side_effect = [
        _Stream([_chunk("First one. "), _chunk("Second one.")]),
        _tool_response({"emotions": "happy"}),
    ]
    tts = MagicMock(voice="alloy")
    spoken = []

    async def synth(text):
        spoken.append(text)
        yield b"RIFF" + b"\x00" * 60

    tts.synthesize_stream = synth
    sent = []
    router = mocker.patch("src.core.orchestrator.router")
    router.route_command = AsyncMock(side_effect=lambda cmd: sent.append(cmd.command))
    mocker.patch("src.core.orchestrator.telemetry")

    orch = DialogOrchestrator(
        stt_providers={"s": MagicMock()}, llm_providers={"openai": OpenAILLMProvider()},
        tts_providers={"t": tts}, default_llm="openai", default_tts="t", default_stt="s",
    )
    await orch.process_text_interaction("hi", "vr_headset", "agent_alpha")

    assert spoken == ["First one.", "Second one."]
    assert sent.count("llm_reply_chunk") == 2
    assert sent.count("tts_complete") == 2
    assert "execute_state" in sent
