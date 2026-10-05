"""
Tests for two consistency fixes:

1. The single-value-category rule holds in every tool schema the server builds,
   not only in the main one: a category with one value is offered to the model
   but never required (Gemini actions-only schema, generic OpenAI-compatible
   adapter).
2. The session record keeps what clients send, not only what the server sends:
   speech, typed and operator-injected requests are logged as ``inbound`` events
   (participant audio is replaced by its size), and markers carry their stable id
   so amendments can be joined to them.
"""

import os

os.environ["OVARP_TESTING"] = "1"

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient

import src.main as main_module
from src.core.config import OVARPConfig
from src.core.router import CommandRouter
from src.core.runtime import runtime
from src.core.schemas import BaseCommand
from src.core.session_manager import SessionManager
from src.core.telemetry import telemetry
from src.providers.custom_provider import CustomLLMProvider
from src.providers.gemini_provider import GeminiLLMProvider


@pytest.fixture
def config_with_single_value(mocker):
    cfg = OVARPConfig(**{
        "experiment": {"name": "t", "description": "d", "version": "1"},
        "devices": [{"id": "web_01", "name": "Web", "type": "web"}],
        "agents": [{"id": "agent_alpha", "name": "A", "description": "d"}],
        "custom_commands": {
            "emotions": {"description": "d", "values": ["happy", "sad"]},
            "avatar": {"description": "d", "values": ["default"]},
        },
    })
    mocker.patch("src.core.config.ConfigManager.config",
                 new_callable=PropertyMock, return_value=cfg)
    return cfg


def _required(schema_params):
    return set(schema_params["required"])


class TestSingleValueRuleEverywhere:
    def test_gemini_actions_only_schema(self, config_with_single_value):
        schema = GeminiLLMProvider()._build_actions_only_tools_schema()
        params = schema[0]["function_declarations"][0]["parameters"]
        assert "avatar" in params["properties"]
        assert "avatar" not in _required(params)
        assert "emotions" in _required(params)

    def test_gemini_main_schema_unchanged(self, config_with_single_value):
        schema = GeminiLLMProvider()._build_tools_schema()
        params = schema[0]["function_declarations"][0]["parameters"]
        assert "avatar" not in _required(params)
        assert {"emotions", "spoken_response"} <= _required(params)

    def test_custom_provider_schema(self, config_with_single_value):
        provider = CustomLLMProvider(name="local", base_url="http://localhost:11434/v1", model="m")
        params = provider._build_tools_schema()[0]["function"]["parameters"]
        assert "avatar" in params["properties"]
        assert "avatar" not in _required(params)
        assert {"emotions", "spoken_response"} <= _required(params)


class FakeFileLogger:
    def __init__(self):
        self.entries = []

    def info(self, **kwargs):
        self.entries.append(kwargs)


@pytest.fixture
def captured(monkeypatch):
    fake = FakeFileLogger()
    monkeypatch.setattr(telemetry, "file_logger", fake)
    return fake


class TestInboundRecorded:
    def test_audio_payload_replaced_by_size(self, captured):
        cmd = BaseCommand(sender="web_01", target_device="all", target_agent="agent_alpha",
                          command_type="audio", command="stt_request",
                          subcommand={"audio_base64": "AAAA" * 10})
        telemetry.log_inbound(cmd)
        entry = captured.entries[-1]
        assert entry["event"] == "inbound"
        assert "audio_base64" not in entry["subcommand"]
        assert entry["subcommand"]["audio_bytes"] == 30
        # the original command is not mutated
        assert "audio_base64" in cmd.subcommand

    @pytest.mark.parametrize("ctype,cmd_name,sub", [
        ("message", "llm_request", {"text": "hello there"}),
        ("message", "direct_tts", {"text": "wizard line"}),
        ("audio", "stt_request", {"audio_base64": "AAAA"}),
    ])
    def test_router_logs_each_request_kind(self, captured, ctype, cmd_name, sub):
        router = CommandRouter()
        router.set_orchestrator(AsyncMock())
        cmd = BaseCommand(sender="web_01", target_device="all", target_agent="agent_alpha",
                          command_type=ctype, command=cmd_name, subcommand=sub)

        async def go():
            await router.route_command(cmd)
            await asyncio.sleep(0)

        asyncio.run(go())
        inbound = [e for e in captured.entries if e.get("event") == "inbound"]
        assert len(inbound) == 1
        assert inbound[0]["command"] == cmd_name
        if "text" in sub:
            assert inbound[0]["subcommand"]["text"] == sub["text"]


class TestMarkerIdRecorded:
    def test_api_marker_logs_its_id(self, monkeypatch):
        mgr = SessionManager()
        mgr._session = None
        mgr._last_completed = None
        mock_tel = MagicMock()
        monkeypatch.setattr(runtime, "session_manager", mgr, raising=False)
        monkeypatch.setattr(runtime, "telemetry", mock_tel, raising=False)
        mgr.start_session("P001")

        client = TestClient(main_module.app)
        resp = client.post("/api/session/marker", json={"label": "Task Started"})
        assert resp.status_code == 200
        marker_id = resp.json()["marker"]["id"]
        kwargs = mock_tel.log_marker.call_args.kwargs
        assert kwargs["marker_id"] == marker_id

    def test_log_marker_writes_id(self, captured):
        telemetry.log_marker("Task Started", {"source": "x"}, marker_id="abc12345", category="task")
        entry = captured.entries[-1]
        assert entry["event"] == "marker"
        assert entry["marker_id"] == "abc12345"
        assert entry["category"] == "task"


# ---------------------------------------------------------------------------
# Sentence-level path: gestures go out while speech is still being produced,
# and a failed stream is reported instead of ending the turn silently.
# ---------------------------------------------------------------------------
from src.core.orchestrator import DialogOrchestrator  # noqa: E402
from src.providers.base import BaseSTTProvider, BaseTTSProvider  # noqa: E402


def _streaming_orchestrator(stream, actions=None, tts_delay=0.0):
    stt = MagicMock(spec=BaseSTTProvider)
    llm = MagicMock()
    llm.model = "stream-model"
    llm.stream_reply = stream
    llm.extract_actions = AsyncMock(return_value=actions or {})
    tts = MagicMock(spec=BaseTTSProvider)
    tts.voice = "v"

    async def synth(text):
        await asyncio.sleep(tts_delay)
        yield b"RIFF" + b"\x00" * 40

    tts.synthesize_stream = synth
    return DialogOrchestrator(stt_provider=stt, llm_providers={"g": llm},
                              tts_providers={"g": tts}, default_llm="g", default_tts="g")


class TestSentenceLevelPath:
    def test_gestures_dispatched_before_last_audio(self, mocker):
        async def stream(prompt, system_prompt=None, history=None):
            for part in ["One. ", "Two. ", "Three. ", "Four."]:
                yield part

        orch = _streaming_orchestrator(stream, actions={"emotions": "happy"}, tts_delay=0.05)
        sent = []
        mock_router = mocker.patch("src.core.orchestrator.router")

        async def record(cmd):
            sent.append(cmd.command)

        mock_router.route_command = AsyncMock(side_effect=record)
        asyncio.run(orch.process_text_interaction("hi", "all", "agent_alpha"))
        assert "execute_state" in sent
        last_audio = max(i for i, c in enumerate(sent) if c == "tts_complete")
        assert sent.index("execute_state") < last_audio

    def test_stream_failure_reports_pipeline_error(self, mocker):
        async def stream(prompt, system_prompt=None, history=None):
            yield "First sentence. "
            raise RuntimeError("provider went away")

        orch = _streaming_orchestrator(stream)
        mock_router = mocker.patch("src.core.orchestrator.router")
        mock_router.route_command = AsyncMock()
        mock_router.dispatch_outbound = AsyncMock()
        asyncio.run(orch.process_text_interaction("hi", "all", "agent_alpha"))
        errors = [c.args[0] for c in mock_router.dispatch_outbound.call_args_list
                  if c.args[0].command == "pipeline_error"]
        assert errors and errors[0].subcommand["stage"] == "llm"


class TestDeclaredCategoriesOnly:
    def test_unknown_category_key_is_rejected(self):
        with pytest.raises(ValueError):
            BaseCommand(sender="woz", target_device="all", target_agent="agent_alpha",
                        command_type="action", command="execute_state",
                        subcommand={"emotoins": "happy"})

    def test_declared_category_accepted(self):
        cmd = BaseCommand(sender="woz", target_device="all", target_agent="agent_alpha",
                          command_type="action", command="execute_state",
                          subcommand={"emotions": "happy"})
        assert cmd.subcommand == {"emotions": "happy"}


class TestOpenAIModelsConfigurable:
    def test_env_vars_select_models(self, monkeypatch):
        from src.providers.openai_provider import (
            OpenAILLMProvider,
            OpenAISTTProvider,
            OpenAITTSProvider,
        )
        monkeypatch.setenv("OVARP_OPENAI_LLM_MODEL", "gpt-x")
        monkeypatch.setenv("OVARP_OPENAI_TTS_MODEL", "tts-x")
        monkeypatch.setenv("OVARP_OPENAI_STT_MODEL", "stt-x")
        assert OpenAILLMProvider().model == "gpt-x"
        assert OpenAITTSProvider().model == "tts-x"
        assert OpenAISTTProvider().model == "stt-x"

    def test_defaults_unchanged(self, monkeypatch):
        from src.providers.openai_provider import OpenAILLMProvider
        monkeypatch.delenv("OVARP_OPENAI_LLM_MODEL", raising=False)
        assert OpenAILLMProvider().model == "gpt-4o"
