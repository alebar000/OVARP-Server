"""The session record keeps synthesized speech as WAV files, written off the
event loop, and audio chunks are serialized without re-scanning their base64.

These pin the properties the latency work relies on: the record stays complete
(every utterance's audio, its text and duration, in timeline order), and the
fast serialization path produces exactly the JSON the normal path would.
"""

import asyncio
import json
import os
import struct
from unittest.mock import AsyncMock

import pytest

os.environ["OVARP_TESTING"] = "1"

from src.core import telemetry as telemetry_module  # noqa: E402
from src.core.orchestrator import DialogOrchestrator  # noqa: E402
from src.core.schemas import Base64Text, BaseCommand, encode_audio  # noqa: E402
from src.core.telemetry import TelemetryLogger, _RecordWriter  # noqa: E402
from src.providers.base import BaseLLMProvider, BaseSTTProvider, BaseTTSProvider  # noqa: E402


def _wav(n_bytes: int, rate: int = 24000) -> bytes:
    header = (b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
              + b"data" + struct.pack("<I", n_bytes))
    return header + b"\x01\x02" * (n_bytes // 2)


def _logger(tmp_path) -> TelemetryLogger:
    """A TelemetryLogger on a temp file, without touching the global structlog config."""
    logger = TelemetryLogger.__new__(TelemetryLogger)
    logger.log_dir = tmp_path
    logger.session_id = "t"
    logger.jsonl_path = tmp_path / "session_t.jsonl"
    logger._writer = _RecordWriter(logger.jsonl_path)
    logger.file_logger = logger._writer
    logger._audio_seq = 0
    return logger


def _records(logger) -> list[dict]:
    assert logger.flush()
    return [json.loads(line) for line in logger.jsonl_path.read_text().splitlines() if line]


def _cmd(subcommand, command="tts_chunk", command_type="audio") -> BaseCommand:
    return BaseCommand(sender="server_orchestrator", target_device="all",
                       target_agent="agent_alpha", command_type=command_type,
                       command=command, subcommand=subcommand)


class TestAudioWire:
    def test_fast_path_matches_pydantic_output(self):
        cmd = _cmd({"audio_base64": encode_audio(_wav(4000))})
        assert type(cmd.subcommand["audio_base64"]) is Base64Text
        assert cmd.to_json() == cmd.model_dump_json()
        sent = json.loads(cmd.to_json())["subcommand"]["audio_base64"]
        assert sent == cmd.subcommand["audio_base64"]

    def test_client_text_takes_the_normal_path(self):
        # A plain str is never spliced, even if it looks like base64.
        cmd = _cmd({"audio_base64": 'AAAA"}, "x": "y'})
        assert json.loads(cmd.to_json())["subcommand"]["audio_base64"] == 'AAAA"}, "x": "y'

    def test_other_commands_unchanged(self):
        cmd = _cmd({"text": "hi", "agent": "agent_alpha"},
                   command="llm_reply", command_type="message")
        assert cmd.to_json() == cmd.model_dump_json()


class TestRecord:
    def test_chunk_recorded_by_size_not_content(self, tmp_path):
        logger = _logger(tmp_path)
        data = _wav(1000)
        logger.log_interaction(_cmd({"audio_base64": encode_audio(data)}))
        entry = _records(logger)[-1]
        assert entry["command"] == "tts_chunk"
        assert entry["subcommand"] == {"audio_bytes": len(data)}

    def test_utterance_saved_as_wav_with_text_and_duration(self, tmp_path):
        logger = _logger(tmp_path)
        data = _wav(48000)  # one second at 24 kHz, 16-bit mono
        chunks = [data[:44], data[44:20000], data[20000:]]
        rel = logger.log_agent_audio(chunks, "Hello there.", "agent_alpha")
        entry = _records(logger)[-1]
        assert entry["event"] == "agent_audio"
        assert entry["text"] == "Hello there."
        assert entry["audio_bytes"] == len(data)
        assert entry["duration_ms"] == 1000
        assert (tmp_path / rel).read_bytes() == data

    def test_records_keep_submission_order(self, tmp_path):
        logger = _logger(tmp_path)
        for i in range(500):
            logger.file_logger.info(event="interaction", seq=i)
        assert [r["seq"] for r in _records(logger)] == list(range(500))

    def test_marker_payload_is_captured_when_logged(self, tmp_path):
        # Later edits to the metadata object must not reach the record.
        logger = _logger(tmp_path)
        meta = {"source": "console"}
        logger.log_marker("Task Started", meta, marker_id="m1")
        meta["source"] = "edited later"
        assert _records(logger)[-1]["marker_metadata"] == {"source": "console"}


class _StreamingLLM(BaseLLMProvider):
    model = "stream"

    async def generate_response(self, prompt, system_prompt=None):  # pragma: no cover
        yield ""

    async def generate_response_with_actions(self, prompt, system_prompt=None, history=None):
        return "One. Two.", {}

    async def stream_reply(self, prompt, system_prompt=None, history=None):
        for word in ["One. ", "Two."]:
            await asyncio.sleep(0)
            yield word

    async def extract_actions(self, spoken_text, system_prompt=None):
        return {}


class _TTS(BaseTTSProvider):
    model = "tts"

    def __init__(self):
        self.voice = "v"

    async def synthesize_stream(self, text):
        data = _wav(4800)
        yield data[:44]
        yield data[44:]


class _STT(BaseSTTProvider):
    async def transcribe(self, audio_data):  # pragma: no cover
        return ""


def _orchestrator() -> DialogOrchestrator:
    return DialogOrchestrator(
        stt_providers={"s": _STT()}, llm_providers={"l": _StreamingLLM()},
        tts_providers={"t": _TTS()}, default_llm="l", default_tts="t", default_stt="s",
    )


def test_each_sentence_is_saved_with_its_text(monkeypatch):
    saved = []
    monkeypatch.setattr(telemetry_module.telemetry, "log_agent_audio",
                        lambda chunks, text, agent: saved.append((text, sum(map(len, chunks)))))
    orch = _orchestrator()
    monkeypatch.setattr("src.core.orchestrator.router.route_command", AsyncMock())
    asyncio.run(orch.process_text_interaction("hi", "web_01", "agent_alpha"))
    assert saved == [("One.", len(_wav(4800))), ("Two.", len(_wav(4800)))]


@pytest.mark.asyncio
async def test_full_reply_audio_is_saved(monkeypatch):
    saved = []
    monkeypatch.setattr(telemetry_module.telemetry, "log_agent_audio",
                        lambda chunks, text, agent: saved.append((text, b"".join(chunks))))
    orch = _orchestrator()
    monkeypatch.setattr("src.core.orchestrator.router.route_command", AsyncMock())
    await orch._dispatch_tts("Wizard line.", "web_01", "agent_alpha")
    assert saved == [("Wizard line.", _wav(4800))]
