"""OpenAI text-to-speech through the Realtime API.

OpenAI retires its speech-endpoint models on 2027-01-06 and names
gpt-realtime-2.1-mini as the replacement. A Realtime model converses, so the
adapter instructs it to read verbatim and checks every utterance against the
transcript of what it said. These tests drive the adapter with a fake connection.
"""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.providers.openai_provider import READER_INSTRUCTIONS, OpenAITTSProvider


class FakeConnection:
    """Answers each response.create with audio deltas, a transcript and response.done."""

    def __init__(self, pcm_per_response=b"\x01\x00" * 480, transcript=None, fail=None,
                 status="completed"):
        self.sent = []
        self.closed = False
        self._pcm = pcm_per_response
        self._transcript = transcript
        self._fail = fail
        self._status = status
        self._events = []
        self.session = SimpleNamespace(update=self._session_update)
        self.response = SimpleNamespace(create=self._response_create)

    async def _session_update(self, session):
        self.sent.append(("session.update", session))

    async def _response_create(self, response):
        self.sent.append(("response.create", response))
        if self._fail:
            raise self._fail
        text = response["input"][0]["content"][0]["text"]
        half = len(self._pcm) // 2
        for part in (self._pcm[:half], self._pcm[half:]):
            self._events.append(SimpleNamespace(type="response.output_audio.delta",
                                                delta=base64.b64encode(part).decode()))
        self._events.append(SimpleNamespace(type="response.output_audio_transcript.done",
                                            transcript=self._transcript or text))
        self._events.append(SimpleNamespace(type="response.done",
                                            response=SimpleNamespace(status=self._status,
                                                                     status_details=None)))

    async def recv(self):
        return self._events.pop(0)

    async def close(self):
        self.closed = True


@pytest.fixture
def realtime(mocker):
    """Patches the OpenAI client; ``opened`` lists every connection the adapter opens."""
    opened = []
    factory = {"make": lambda: FakeConnection()}

    class Manager:
        async def enter(self):
            conn = factory["make"]()
            opened.append(conn)
            return conn

    client = MagicMock()
    client.realtime.connect = MagicMock(side_effect=lambda model: Manager())
    mocker.patch("src.providers.openai_provider.OpenAIClientSingleton.get_client",
                 return_value=client)
    return SimpleNamespace(client=client, opened=opened, factory=factory)


async def _speak(tts, text):
    return b"".join([c async for c in tts.synthesize_stream(text)])


def test_default_is_the_realtime_replacement(monkeypatch):
    monkeypatch.delenv("OVARP_OPENAI_TTS_MODEL", raising=False)
    tts = OpenAITTSProvider()
    assert tts.model == "gpt-realtime-2.1-mini" and tts.uses_realtime


@pytest.mark.asyncio
async def test_reads_text_out_of_band_and_returns_a_wav(realtime):
    tts = OpenAITTSProvider(voice="marin")
    audio = await _speak(tts, "Hola, ¿cómo estás?")

    assert audio[:4] == b"RIFF" and audio[22:24] == b"\x01\x00"  # mono
    assert int.from_bytes(audio[24:28], "little") == 24000
    assert audio[44:] == b"\x01\x00" * 480

    kind, session = realtime.opened[0].sent[0]
    assert kind == "session.update"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000},
                                          "voice": "marin"}
    kind, response = realtime.opened[0].sent[1]
    assert response["conversation"] == "none"  # nothing accumulates between utterances
    assert response["instructions"] == READER_INSTRUCTIONS
    assert response["input"][0]["content"][0]["text"] == "Hola, ¿cómo estás?"


@pytest.mark.asyncio
async def test_connection_is_reused_per_voice(realtime):
    tts = OpenAITTSProvider(voice="cedar")
    await _speak(tts, "One.")
    await _speak(tts, "Two.")
    assert len(realtime.opened) == 1
    tts.voice = "marin"  # a session's voice is fixed once it has spoken
    await _speak(tts, "Three.")
    assert len(realtime.opened) == 2


@pytest.mark.asyncio
async def test_a_deviation_from_the_text_is_recorded(realtime, mocker):
    realtime.factory["make"] = lambda: FakeConnection(transcript="Hello, how can I help you?")
    record = mocker.patch("src.providers.openai_provider.logger")
    await _speak(OpenAITTSProvider(), "Hello.")
    record.warning.assert_called_once()
    assert record.warning.call_args.kwargs["requested"] == "Hello."
    assert record.warning.call_args.kwargs["spoken"] == "Hello, how can I help you?"


@pytest.mark.asyncio
async def test_punctuation_case_and_accents_are_not_deviations(realtime, mocker):
    realtime.factory["make"] = lambda: FakeConnection(transcript="hola como estas")
    record = mocker.patch("src.providers.openai_provider.logger")
    await _speak(OpenAITTSProvider(), "Hola, ¿cómo estás?")
    record.warning.assert_not_called()


@pytest.mark.asyncio
async def test_a_stale_pooled_connection_is_replaced(realtime):
    tts = OpenAITTSProvider()
    await _speak(tts, "First.")
    stale = realtime.opened[0]
    stale._fail = ConnectionError("session expired")
    assert await _speak(tts, "Second.")
    assert stale.closed and len(realtime.opened) == 2


@pytest.mark.asyncio
async def test_a_failed_response_raises(realtime):
    realtime.factory["make"] = lambda: FakeConnection(status="failed")
    with pytest.raises(RuntimeError):
        await _speak(OpenAITTSProvider(), "Hi.")
    assert realtime.opened[0].closed


@pytest.mark.asyncio
async def test_legacy_voices_map_to_realtime_voices(realtime):
    await _speak(OpenAITTSProvider(voice="nova"), "Hi.")
    assert realtime.opened[0].sent[0][1]["audio"]["output"]["voice"] == "marin"
