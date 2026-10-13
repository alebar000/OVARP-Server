"""
Open Virtual Agent Research Platform (OVARP) — OpenAI Provider

Implements STT, LLM, and TTS providers using the OpenAI API:
- ``OpenAISTTProvider``: Whisper-based speech-to-text transcription.
- ``OpenAILLMProvider``: GPT chat completions with tool/function calling
  for agent actions (emotions, gestures, gaze).
- ``OpenAITTSProvider``: Text-to-speech synthesis using the ``tts-1`` model.

Author: Alexander Barquero Elizondo, Ph.D. — UCR, ECCI/CITIC
License: MIT
"""

import base64
import os
import json
import re
import struct
import unicodedata
from typing import AsyncGenerator, Dict, Any, Optional
from openai import AsyncOpenAI
import logging
from structlog import get_logger

from src.providers.base import BaseSTTProvider, BaseLLMProvider, BaseTTSProvider
from src.core.config import config_manager

logger = get_logger()
std_log = logging.getLogger("OVARP.openai")

# The cheapest current models that are good enough for a conversational agent.
# Each can be changed through its OVARP_OPENAI_*_MODEL variable.
DEFAULT_LLM_MODEL = "gpt-6-luna"
DEFAULT_STT_MODEL = "gpt-transcribe"
# OpenAI's named replacement for its speech-endpoint models (tts-1, gpt-4o-mini-tts),
# which are retired on 2027-01-06. Served by the Realtime API; see OpenAITTSProvider.
DEFAULT_TTS_MODEL = "gpt-realtime-2.1-mini"


def reasoning_kwargs() -> Dict[str, Any]:
    """``reasoning_effort`` for the LLM calls, from OVARP_OPENAI_REASONING_EFFORT.

    Defaults to ``none``: a conversational turn should not wait for reasoning, and
    on Chat Completions the current models accept function calling only without
    it. Set the variable to an empty value for models that do not take the
    parameter.
    """
    effort = os.getenv("OVARP_OPENAI_REASONING_EFFORT", "none").strip()
    return {"reasoning_effort": effort} if effort else {}

class OpenAIClientSingleton:
    """Manages the shared AsyncOpenAI client instance."""
    _instance = None
    _client: AsyncOpenAI = None

    @classmethod
    def get_client(cls) -> AsyncOpenAI:
        if cls._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY environment variable not set. OpenAI won't authenticate.")
                # We instantiate with a dummy key to prevent immediate pydantic validation crashes at boot,
                # though any actual API call will fail gracefully later.
                cls._client = AsyncOpenAI(api_key="sk-dummy")
            else:
                cls._client = AsyncOpenAI(api_key=api_key)
        return cls._client

    @classmethod
    def reset_client(cls):
        """Drop the cached client so the next call picks up a changed API key."""
        cls._client = None

class OpenAISTTProvider(BaseSTTProvider):
    """Whisper transcription provider."""
    def __init__(self, model_name: str = None):
        self.model = model_name or os.getenv("OVARP_OPENAI_STT_MODEL", DEFAULT_STT_MODEL)

    @property
    def client(self) -> AsyncOpenAI:
        return OpenAIClientSingleton.get_client()

    async def transcribe(self, audio_data: bytes) -> str:
        # Note: OpenAI expects a named file or a tuple (filename, file_content)
        # Ideally, we receive WAV or similar encoded binary audio from the XR client
        try:
            std_log.info(f"🎤 STT: Starting transcription | audio_size={len(audio_data)} bytes")
            file_tuple = ("audio.wav", audio_data, "audio/wav")
            transcription = await self.client.audio.transcriptions.create(
                model=self.model,
                file=file_tuple,
                response_format="text"
            )
            std_log.info(f"✅ STT: Transcription complete | text=\"{str(transcription)[:80]}\"")
            return transcription
        except Exception as e:
            std_log.error(f"❌ STT: Transcription failed | {type(e).__name__}: {str(e)}")
            logger.error("STT transcription failed", error=str(e))
            return ""

# ---------------------------------------------------------------------------
# Optional streaming capability, shared by every OpenAI-compatible adapter.
#
# The orchestrator probes for ``stream_reply`` / ``extract_actions``. With them it
# speaks sentence by sentence while the reply is still being generated, and
# classifies gestures and emotion in a separate call over the finished text, off
# the audio path. Text is streamed without tools: a forced tool call would make
# the model hold back its reply until the call's arguments are complete.
# ---------------------------------------------------------------------------

PLAIN_TEXT_OVERRIDE = (
    "\nFor this reply, respond with plain spoken text only. Do not call any function or tool."
)

CLASSIFY_PROMPT = (
    "You just replied with the following text to the user:\n\"{text}\"\n\n"
    "Call update_agent_state to classify your emotional/physical state for that reply. "
    "Do not repeat or alter the text."
)

NAMED_TOOL_CHOICE = {"type": "function", "function": {"name": "update_agent_state"}}


def chat_messages(prompt: str, system_prompt: Optional[str], history: Optional[list],
                  plain_text: bool = False) -> list[dict]:
    """Messages for a chat completion. ``plain_text`` cancels, for this call only,
    an instruction in the agent's prompt to always call ``update_agent_state``."""
    messages = []
    if system_prompt:
        content = system_prompt + (PLAIN_TEXT_OVERRIDE if plain_text else "")
        messages.append({"role": "system", "content": content})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def actions_only_tools() -> list[Dict[str, Any]]:
    """The ``update_agent_state`` tool without ``spoken_response``: it classifies a
    reply that already exists instead of asking the model to write it again."""
    properties = {}
    required = []
    for cat_name, category in config_manager.config.custom_commands.items():
        properties[cat_name] = {
            "type": "string",
            "enum": category.values,
            "description": category.description,
        }
        # A single-value category is offered but not demanded (same rule as the
        # full tool schema), so the model is never forced to restate it.
        if len(category.values) > 1:
            required.append(cat_name)
    return [{
        "type": "function",
        "function": {
            "name": "update_agent_state",
            "description": (
                "Classify the agent's state (emotion/action) for the reply you just gave."
            ),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }]


async def stream_chat_text(client: AsyncOpenAI, model: str, prompt: str,
                           system_prompt: Optional[str], history: Optional[list],
                           **extra: Any) -> AsyncGenerator[str, None]:
    """Yield the reply's text as the endpoint generates it. Errors propagate, so
    the orchestrator reports a ``pipeline_error`` instead of ending in silence."""
    stream = await client.chat.completions.create(
        model=model,
        messages=chat_messages(prompt, system_prompt, history, plain_text=True),
        stream=True,
        **extra,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue  # e.g. a trailing usage-only chunk
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def classify_actions(client: AsyncOpenAI, model: str, spoken_text: str,
                           system_prompt: Optional[str], tool_choice: Any = None,
                           **extra: Any) -> Dict[str, Any]:
    """Lean follow-up call: the expressive state for a reply already spoken.

    Returns ``{}`` when nothing is declared to classify. Raises on API errors so
    each adapter decides how to degrade.
    """
    tools = actions_only_tools()
    if not spoken_text or not tools[0]["function"]["parameters"]["properties"]:
        return {}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": CLASSIFY_PROMPT.format(text=spoken_text)})
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice or NAMED_TOOL_CHOICE,
        **extra,
    )
    message = response.choices[0].message
    for tool_call in message.tool_calls or []:
        if tool_call.function.name == "update_agent_state":
            args = json.loads(tool_call.function.arguments or "{}")
            args.pop("spoken_response", None)
            return args
    return {}


class OpenAILLMProvider(BaseLLMProvider):
    """GPT-based Language Model using Tool Calling for dynamic configuration actions."""
    def __init__(self, model_name: str = None):
        self.model = model_name or os.getenv("OVARP_OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL)

    @property
    def client(self) -> AsyncOpenAI:
        return OpenAIClientSingleton.get_client()

    def _build_tools_schema(self) -> list[Dict[str, Any]]:
        """Dynamically build OpenAI Tool Schema from the config_manager."""
        config = config_manager.config
        
        properties = {}
        required = []

        # Iterate all custom defined commands from YAML config (ex: emotions, actions)
        for cat_name, category in config.custom_commands.items():
            properties[cat_name] = {
                "type": "string",
                "enum": category.values,
                "description": category.description
            }
            # A category with a single value leaves the model nothing to decide, so
            # it is offered but not demanded. Requiring it made the model restate
            # the only option on every turn, which the console then drew as a tag
            # on every reply. Add a second value and it becomes required again.
            if len(category.values) > 1:
                required.append(cat_name)

        # Add spoken_response as a required param so the LLM always provides text
        properties["spoken_response"] = {
            "type": "string",
            "description": "Your spoken reply to the user. This is what the agent will say out loud."
        }
        required.append("spoken_response")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_agent_state",
                    "description": "Always call this function with your spoken reply and the agent's updated state.",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        ]
        return tools

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **reasoning_kwargs(),
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_response_with_actions(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[list] = None) -> tuple[str, Dict[str, Any]]:
        """Non-streaming call that returns text and parallel tool arguments (JSON)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Include conversation history for multi-turn awareness
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        tools = self._build_tools_schema()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="required",  # Force the model to always call update_agent_state
            **reasoning_kwargs(),
        )

        message = response.choices[0].message
        spoken_text = message.content or ""
        
        actions = {}
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "update_agent_state":
                    args = json.loads(tool_call.function.arguments)
                    # Extract spoken_response from function args if text content is empty
                    if not spoken_text and "spoken_response" in args:
                        spoken_text = args.pop("spoken_response")
                    else:
                        args.pop("spoken_response", None)
                    actions = args

        return spoken_text, actions

    async def stream_reply(self, prompt: str, system_prompt: Optional[str] = None,
                           history: Optional[list] = None) -> AsyncGenerator[str, None]:
        """Incremental text of the spoken reply (optional streaming capability)."""
        std_log.info(f"🔄 OpenAI: Streaming text call | model={self.model} | prompt=\"{prompt[:80]}\"")
        try:
            async for delta in stream_chat_text(self.client, self.model, prompt,
                                                system_prompt, history, **reasoning_kwargs()):
                yield delta
        except Exception as e:
            std_log.error(f"❌ OpenAI: Streaming text call FAILED | {type(e).__name__}: {e}")
            raise

    async def extract_actions(self, spoken_text: str,
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gestures and emotion for a reply that was already streamed."""
        try:
            return await classify_actions(self.client, self.model, spoken_text, system_prompt,
                                          **reasoning_kwargs())
        except Exception as e:
            std_log.warning(f"⚠️ OpenAI: extract_actions call failed | {type(e).__name__}: {e}")
            return {}

def _wav_header(data_size: int, sample_rate: int = 24000) -> bytes:
    """44-byte header for mono 16-bit PCM."""
    return (b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", data_size))


def _normalized_words(text: str) -> list[str]:
    """Words compared by the verbatim check: case, accents and punctuation are
    not differences in what was said."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", plain).split()


# Speech-endpoint models (/v1/audio/speech), retired by OpenAI on 2027-01-06.
_SPEECH_ENDPOINT_PREFIXES = ("tts-1", "gpt-4o-mini-tts", "gpt-4o-tts")

# Voices of the speech endpoint that the Realtime API does not have, mapped to the
# closest Realtime voice so existing profiles keep working.
_REALTIME_VOICE_FALLBACK = {"nova": "marin", "onyx": "cedar", "fable": "sage"}

READER_INSTRUCTIONS = (
    "You are a text-to-speech reader, not a conversational assistant. Read the user's "
    "message aloud exactly as written, word for word, in the language it is written in. "
    "Do not answer it, comment on it, translate it, or add, drop or change any word. "
    "Speak naturally, with the punctuation's pauses and intonation."
)


class OpenAITTSProvider(BaseTTSProvider):
    """OpenAI text-to-speech.

    The default model, ``gpt-realtime-2.1-mini``, is served by the Realtime API,
    which OpenAI names as the replacement for its speech-endpoint models. A
    Realtime model is a conversational model, so it is instructed to read the
    text verbatim, and each utterance is checked: the transcript of what it
    actually said is compared with the text, and a deviation is logged to the
    session record (``TTS deviation``) and the server log. Connections are kept
    open and reused, one synthesis at a time each, per voice.

    Speech-endpoint models (``tts-1``, ``gpt-4o-mini-tts``) still work until
    their retirement by setting ``OVARP_OPENAI_TTS_MODEL``.
    """

    def __init__(self, model_name: str = None, voice: str = "alloy"):
        self.model = model_name or os.getenv("OVARP_OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
        self.voice = voice
        self._idle: dict[str, list] = {}
        self._pool_client = None

    @property
    def client(self) -> AsyncOpenAI:
        return OpenAIClientSingleton.get_client()

    @property
    def uses_realtime(self) -> bool:
        return not self.model.startswith(_SPEECH_ENDPOINT_PREFIXES)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        std_log.info(f"🔊 TTS: Starting synthesis | model={self.model} voice={self.voice} "
                     f"text=\"{text[:60]}\"")
        if self.uses_realtime:
            pcm = await self._synthesize_realtime(text)
            audio_bytes = _wav_header(len(pcm)) + pcm
        else:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="wav",  # WAV allows easier playback in Unity and browsers
            )
            # .content, not iter_bytes(): on AsyncOpenAI the latter is a sync iterator
            # and 'async for' over it silently yields nothing.
            audio_bytes = response.content
        if not audio_bytes:
            raise RuntimeError(f"OpenAI TTS ({self.model}) returned no audio")
        std_log.info(f"✅ TTS: Audio received | size={len(audio_bytes)} bytes")

        # Yield in chunks for streaming over ZMQ/WS
        chunk_size = 32 * 1024
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]

    # -- Realtime path -------------------------------------------------------------
    def _realtime_voice(self) -> str:
        voice = _REALTIME_VOICE_FALLBACK.get(self.voice, self.voice)
        if voice != self.voice:
            std_log.warning(f"⚠️ TTS: voice '{self.voice}' is not offered by the Realtime API; "
                            f"using '{voice}'")
        return voice

    async def _acquire(self, voice: str):
        client = self.client
        if client is not self._pool_client:
            # The API key changed: connections opened with the old one are dropped.
            await self.close()
            self._pool_client = client
        idle = self._idle.setdefault(voice, [])
        if idle:
            return idle.pop(), True
        conn = await client.realtime.connect(model=self.model).enter()
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": READER_INSTRUCTIONS,
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000},
                                 "voice": voice}},
        })
        return conn, False

    async def _synthesize_realtime(self, text: str) -> bytes:
        voice = self._realtime_voice()
        for attempt in (1, 2):
            conn, reused = await self._acquire(voice)
            try:
                pcm, transcript = await self._read_aloud(conn, text)
            except Exception:
                await _close_quietly(conn)
                # A pooled connection may have been closed by the server (session
                # limit, idle timeout): retry once on a fresh one.
                if reused and attempt == 1:
                    continue
                raise
            self._idle.setdefault(voice, []).append(conn)
            self._check_verbatim(text, transcript)
            return pcm
        raise RuntimeError("unreachable")

    async def _read_aloud(self, conn, text: str) -> tuple[bytes, str]:
        await conn.response.create(response={
            "conversation": "none",  # out of band: nothing accumulates between utterances
            "output_modalities": ["audio"],
            "instructions": READER_INSTRUCTIONS,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": text}]}],
        })
        pcm = bytearray()
        transcript = ""
        while True:
            event = await conn.recv()
            kind = getattr(event, "type", "")
            if kind == "response.output_audio.delta":
                pcm.extend(base64.b64decode(event.delta))
            elif kind == "response.output_audio_transcript.done":
                transcript = event.transcript or ""
            elif kind == "error":
                detail = getattr(getattr(event, "error", None), "message", "") or str(event)
                raise RuntimeError(f"OpenAI Realtime error: {detail}")
            elif kind == "response.done":
                status = getattr(event.response, "status", "completed")
                if status != "completed":
                    raise RuntimeError(f"OpenAI Realtime response {status}: "
                                       f"{getattr(event.response, 'status_details', None)}")
                return bytes(pcm), transcript

    def _check_verbatim(self, text: str, transcript: str) -> None:
        if _normalized_words(transcript) == _normalized_words(text):
            return
        std_log.warning(f"⚠️ TTS: spoken text differs from the requested text | "
                        f"requested=\"{text[:120]}\" spoken=\"{transcript[:120]}\"")
        logger.warning("TTS deviation", model=self.model, requested=text, spoken=transcript)

    async def close(self) -> None:
        """Close every pooled Realtime connection."""
        for conns in self._idle.values():
            for conn in conns:
                await _close_quietly(conn)
        self._idle.clear()


async def _close_quietly(conn) -> None:
    try:
        await conn.close()
    except Exception:
        pass
