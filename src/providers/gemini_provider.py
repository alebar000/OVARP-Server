"""
Open Virtual Agent Research Platform (OVARP) — Google Gemini Provider

Implements LLM and TTS providers using the Google ``genai`` SDK:
- ``GeminiLLMProvider``: Gemini chat completions with function calling for
  agent actions (emotions, gestures, gaze). Uses a shared singleton client.
  ``stream_reply`` + ``extract_actions`` add real incremental text streaming
  (the optional capability, also offered by the OpenAI-compatible adapters):
  forcing the actions tool call blocks streaming, so the spoken
  reply streams on its own and actions are classified in a lean follow-up call.
- ``GeminiTTSProvider``: text-to-speech with prebuilt voices. The Gemini 3.8 TTS
  models are called through the Interactions API; earlier preview models through
  ``generate_content``.

Default models are the cheapest current ones that are good enough for a
conversational agent (``gemini-3.5-flash-lite``, ``gemini-3.8-flash-lite-tts``).
Each can be changed through its ``OVARP_GEMINI_*_MODEL`` variable.
No sampling temperature is set: Google recommends the default for Gemini 3.

Author: Alexander Barquero Elizondo, Ph.D. — UCR, ECCI/CITIC
License: MIT
"""

import os
import asyncio
import base64
import json
import struct
import logging
import threading
from typing import AsyncGenerator, Dict, Any, Optional
from google import genai
from google.genai import types as genai_types
from structlog import get_logger

from src.providers.base import BaseSTTProvider, BaseLLMProvider, BaseTTSProvider, tts_timeout_s
from src.core.config import config_manager

logger = get_logger()
std_log = logging.getLogger("OVARP.gemini")

# Pinned ids, not "-latest" aliases: an alias can move to another model between
# two sessions of the same study.
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-lite-tts"
# TTS models served through generate_content rather than the Interactions API.
_GENERATE_CONTENT_TTS_PREFIXES = ("gemini-2", "gemini-3.1-flash-tts", "gemini-3-flash-tts")

class GeminiClientSingleton:
    """Manages the shared Gemini client instance."""
    _instance = None
    _client: genai.Client = None

    @classmethod
    def get_client(cls) -> genai.Client:
        if cls._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                std_log.warning("⚠️ Gemini: GEMINI_API_KEY not found in environment. Provider will be disabled.")
                logger.warning("GEMINI_API_KEY environment variable not set.")
                return None
            
            try:
                cls._client = genai.Client(api_key=api_key)
                std_log.info(f"✅ Gemini: Client initialized successfully (key ends in ...{api_key[-4:]})")
            except ValueError as e:
                std_log.error(f"❌ Gemini: Failed to create client | {str(e)}")
                logger.error("Failed to create Gemini client", error=str(e))
                return None

        return cls._client

    @classmethod
    def reset_client(cls):
        """Drop the cached client so the next call picks up a changed API key."""
        cls._client = None

class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini Language Model utilizing Native Tool Calling for configuration actions."""

    # Condition prompts (config.yaml) tell the model to "Always call the
    # update_agent_state function..."; stream_reply doesn't declare that tool, so this
    # cancels the instruction for the streaming call only. See stream_reply's docstring.
    _NO_FUNCTION_CALL_OVERRIDE = (
        "\nFor this reply, respond with plain spoken text only. Do not call any function or tool."
    )

    def __init__(self, model_name: str = None):
        # A dated id goes stale: gemini-2.5-flash is already refused for new
        # accounts. The rolling alias keeps working; the env var is the override.
        self.model = model_name or os.getenv("OVARP_GEMINI_LLM_MODEL", DEFAULT_LLM_MODEL)

    @property
    def client(self) -> genai.Client:
        return GeminiClientSingleton.get_client()

    def _build_tools_schema(self) -> list:
        """Dynamically build Gemini Tool Schema from the config_manager."""
        config = config_manager.config
        
        properties = {}
        required = []

        # Iterate all custom defined commands from YAML config (ex: emotions, actions)
        for cat_name, category in config.custom_commands.items():
            properties[cat_name] = {
                "type": "STRING",
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
            "type": "STRING",
            "description": "Your spoken reply to the user. This is what the agent will say out loud."
        }
        required.append("spoken_response")

        # Gemini expects the JSON Schema format to be slightly different (OpenAPI 3.0 subset)
        # We define a function declaration
        tool_schema = {
            "function_declarations": [
                {
                    "name": "update_agent_state",
                    "description": "Always call this function with your spoken reply and the agent's updated state.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": properties,
                        "required": required
                    }
                }
            ]
        }
        return [tool_schema]

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        # Gemini Python SDK doesn't natively support AsyncGenerator out-of-the-box easily without workarounds
        # For the MVP we await the whole stream generation or loop through the blocking sync iterable
        
        config_kwargs = {}
        if system_prompt:
             config_kwargs["system_instruction"] = system_prompt

        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(**config_kwargs)
        )
        
        # We yield as strings
        for chunk in response:
            if chunk.text:
                 yield chunk.text

    async def generate_response_with_actions(self, prompt: str, system_prompt: Optional[str] = None, history: Optional[list] = None) -> tuple[str, Dict[str, Any]]:
        """Non-streaming call that returns text and parallel tool arguments (JSON)."""
        if not self.client:
            std_log.error("❌ Gemini: Provider called but client is None (missing API key)")
            logger.error("Gemini Provider called but the client is not initialized (missing API key).")
            return "Server Error: Gemini Provider is unconfigured.", {}

        tools = self._build_tools_schema()
        
        config_kwargs = {"tools": tools}
        if system_prompt:
             config_kwargs["system_instruction"] = system_prompt
        
        # Build multi-turn contents from history
        contents = []
        if history:
            for msg in history:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        std_log.info(f"🔄 Gemini: Calling API | model={self.model} | prompt=\"{prompt[:80]}\" | history_turns={len(contents)-1}")
             
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=genai.types.GenerateContentConfig(**config_kwargs)
            )
        except Exception as e:
            std_log.error(f"❌ Gemini: API call FAILED | {type(e).__name__}: {str(e)}")
            logger.error("Gemini API call failed", error=str(e))
            return f"API Error: {str(e)}", {}
        
        spoken_text = ""
        actions = {}
        
        # Manually iterate parts to handle mixed text + function_call responses
        # The SDK's response.text property can fail or return empty when function_calls are present
        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        spoken_text += part.text
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        std_log.info(f"⚡ Gemini: Function call detected | name={fc.name} args={fc.args}")
                        if fc.name == "update_agent_state":
                            args = dict(fc.args)
                            # Extract spoken_response from function args if text is empty
                            if not spoken_text and "spoken_response" in args:
                                spoken_text = args.pop("spoken_response")
                            else:
                                args.pop("spoken_response", None)
                            actions = args
            else:
                # Fallback to simple .text if no candidates structure
                spoken_text = response.text or ""
        except Exception as e:
            std_log.warning(f"⚠️ Gemini: Error parsing response parts | {str(e)}")
            try:
                spoken_text = response.text or ""
            except Exception:
                pass
                    
        std_log.info(f"✅ Gemini: Response complete | text=\"{spoken_text[:80]}\" | actions={actions}")
        return spoken_text, actions

    def _build_actions_only_tools_schema(self) -> list:
        """Same shape as ``_build_tools_schema`` but without ``spoken_response``:
        used by ``extract_actions`` to classify a reply that was already generated,
        instead of asking the model to write it again."""
        config = config_manager.config
        properties = {}
        required = []
        for cat_name, category in config.custom_commands.items():
            properties[cat_name] = {
                "type": "STRING",
                "enum": category.values,
                "description": category.description
            }
            # Same rule as _build_tools_schema: a single-value category is offered
            # but not demanded, so the model is never forced to restate it.
            if len(category.values) > 1:
                required.append(cat_name)
        return [{
            "function_declarations": [
                {
                    "name": "update_agent_state",
                    "description": "Classify the agent's state (emotion/action) for the reply you just gave.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": properties,
                        "required": required
                    }
                }
            ]
        }]

    def _stream_text_sync_to_queue(self, contents: list, config_kwargs: dict, loop: asyncio.AbstractEventLoop,
                                    queue: "asyncio.Queue") -> None:
        """Runs in a worker thread so the blocking network iteration of
        ``generate_content_stream`` doesn't stall the asyncio event loop (which also
        has to keep serving other connected WS/ZMQ clients while this one streams).
        Pushes each text delta, then a sentinel or the exception, back via
        ``call_soon_threadsafe``."""
        SENTINEL = None
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            for chunk in stream:
                if chunk.text:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    async def stream_reply(self, prompt: str, system_prompt: Optional[str] = None,
                            history: Optional[list] = None) -> AsyncGenerator[str, None]:
        """Real, incremental text streaming of the spoken reply.

        Deliberately does NOT force the ``update_agent_state`` tool call here: measured
        2026-08-31 against the live API that with ``tools`` set, ``generate_content_stream``
        buffers the whole function call and delivers it in a single chunk (no incremental
        text/args) -- the SDK's own ``partial_args``/``will_continue`` fields exist for this
        but raise "only supported in Gemini Enterprise Agent" on the plain API key we have.
        So actions can't ride along with a streamed reply; ``extract_actions`` below resolves
        them in a lean follow-up call once the full text is known.

        This project's condition prompts (config.yaml) instruct "Always call the
        update_agent_state function with your spoken reply..." -- measured live
        2026-08-31: with that instruction and no tool declared, Gemini tries to emit a
        function call anyway and returns FinishReason.MALFORMED_FUNCTION_CALL with empty
        text, silently producing an empty reply. ``_NO_FUNCTION_CALL_OVERRIDE`` cancels
        that instruction for this call only.
        """
        if not self.client:
            std_log.error("❌ Gemini: stream_reply called but client is None (missing API key)")
            return

        contents = []
        if history:
            for msg in history:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        config_kwargs = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt + self._NO_FUNCTION_CALL_OVERRIDE

        std_log.info(f"🔄 Gemini: Streaming text call | model={self.model} | prompt=\"{prompt[:80]}\" | history_turns={len(contents)-1}")

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        worker = threading.Thread(
            target=self._stream_text_sync_to_queue,
            args=(contents, config_kwargs, loop, queue),
            daemon=True,
        )
        worker.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                std_log.error(f"❌ Gemini: Streaming text call FAILED | {type(item).__name__}: {str(item)}")
                # Propagate so the orchestrator reports a pipeline_error to the console
                # instead of the turn ending in silence or mid-sentence.
                raise item
            yield item

    async def extract_actions(self, spoken_text: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Lean follow-up call: given a reply that was already generated (and already
        streamed to the client), classify the agent's state (emotion/action enums from
        config.yaml) without regenerating the text. Skipped entirely if no custom
        commands are configured, since there would be nothing to classify."""
        if not self.client or not spoken_text:
            return {}

        tools = self._build_actions_only_tools_schema()
        if not tools[0]["function_declarations"][0]["parameters"]["properties"]:
            return {}

        config_kwargs = {"tools": tools}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        classify_prompt = (
            f"You just replied with the following text to the user:\n\"{spoken_text}\"\n\n"
            "Call update_agent_state to classify your emotional/physical state for that reply. "
            "Do not repeat or alter the text."
        )

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": classify_prompt}]}],
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as e:
            std_log.warning(f"⚠️ Gemini: extract_actions call failed | {type(e).__name__}: {str(e)}")
            return {}

        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name == "update_agent_state":
                        return dict(fc.args)
        except Exception as e:
            std_log.warning(f"⚠️ Gemini: Error parsing extract_actions response | {str(e)}")
        return {}


def _create_wav_header(sample_rate: int, num_channels: int, sample_width: int, data_size: int) -> bytes:
    """Generates a standard 44-byte WAV header for raw PCM data."""
    header = b'RIFF'
    header += struct.pack('<I', 36 + data_size)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16) # Subchunk1Size
    header += struct.pack('<H', 1)  # AudioFormat (1=PCM)
    header += struct.pack('<H', num_channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', sample_rate * num_channels * sample_width) # ByteRate
    header += struct.pack('<H', num_channels * sample_width) # BlockAlign
    header += struct.pack('<H', sample_width * 8) # BitsPerSample
    header += b'data'
    header += struct.pack('<I', data_size)
    return header

class GeminiSTTProvider(BaseSTTProvider):
    """Gemini transcription via generate_content with inline audio.

    Gemini has no dedicated transcription endpoint; audio goes in as an inline
    part alongside an instruction, and the reply is the transcript.
    """

    def __init__(self, model_name: str = None):
        self.model = model_name or os.getenv("OVARP_GEMINI_STT_MODEL", DEFAULT_LLM_MODEL)

    @property
    def client(self) -> genai.Client:
        return GeminiClientSingleton.get_client()

    async def transcribe(self, audio_data: bytes) -> str:
        if not self.client:
            std_log.error("❌ Gemini STT: Client not initialized (missing API key)")
            return ""

        std_log.info(f"🎤 Gemini STT: Starting transcription | audio_size={len(audio_data)} bytes")
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=[
                    genai_types.Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                    "Transcribe this audio verbatim. Reply with the transcript only, "
                    "with no preamble and no quotation marks. If there is no speech, reply "
                    "with nothing at all.",
                ],
            )
            text = (response.text or "").strip()
            std_log.info(f"✅ Gemini STT: Transcription complete | text=\"{text[:80]}\"")
            return text
        except Exception as e:
            std_log.error(f"❌ Gemini STT: Transcription failed | {type(e).__name__}: {str(e)}")
            logger.error("Gemini STT transcription failed", error=str(e))
            return ""


class GeminiTTSProvider(BaseTTSProvider):
    """Gemini text-to-speech with a prebuilt voice.

    Gemini 3.8 TTS models are called through the Interactions API, which returns a
    complete WAV (24 kHz, mono, 16-bit). Earlier preview models use
    ``generate_content_stream`` with the AUDIO modality, which returns raw L16 PCM
    that gets a WAV header here. Either way the audio is complete before it is
    forwarded, so the exact WAV ``data`` size is known and the wire format to
    clients is the same. A failed synthesis raises, so the console reports it
    instead of the agent falling silent.
    """

    def __init__(self, model_name: str = None, voice: str = "Kore"):
        self.model = model_name or os.getenv("OVARP_GEMINI_TTS_MODEL", DEFAULT_TTS_MODEL)
        self.voice = voice

    @property
    def client(self) -> genai.Client:
        return GeminiClientSingleton.get_client()

    @property
    def uses_interactions(self) -> bool:
        return not self.model.startswith(_GENERATE_CONTENT_TTS_PREFIXES)

    async def _synthesize_interactions(self, text: str) -> tuple[bytes, Optional[str]]:
        interaction = await self.client.aio.interactions.create(
            model=self.model,
            input=text,
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": self.voice}]},
        )
        audio = getattr(interaction, "output_audio", None)
        if audio is None or not audio.data:
            return b"", None
        data = audio.data
        raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
        return raw, audio.mime_type or "audio/wav"

    async def _interactions_with_retry(self, text: str) -> tuple[bytes, Optional[str]]:
        """One request, bounded by ``tts_timeout_s``; a request that hangs is
        abandoned and sent once more."""
        timeout = tts_timeout_s(text)
        try:
            return await asyncio.wait_for(self._synthesize_interactions(text), timeout)
        except asyncio.TimeoutError:
            std_log.warning(f"⚠️ Gemini TTS: no audio after {timeout:.0f}s, retrying once "
                            f"| text=\"{text[:60]}\"")
            return await asyncio.wait_for(self._synthesize_interactions(text), timeout)

    def _collect_stream(self, tts_prompt: str) -> tuple[bytes, Optional[str]]:
        """generate_content path (preview models). Runs in a worker thread: consumes
        the streaming response and returns the assembled audio bytes plus the mime
        type of the first audio part seen."""
        audio_bytes = bytearray()
        mime_type = None
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=tts_prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            ),
        )
        for chunk in stream:
            try:
                parts = chunk.candidates[0].content.parts
            except (AttributeError, IndexError, TypeError):
                continue
            if not parts:
                continue
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    audio_bytes.extend(inline.data)
                    if mime_type is None:
                        mime_type = inline.mime_type
        return bytes(audio_bytes), mime_type

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not self.client:
            raise RuntimeError("Gemini TTS client not initialized (missing GEMINI_API_KEY)")

        std_log.info(
            f"🔊 Gemini TTS: Starting synthesis | model={self.model} voice={self.voice} "
            f"text=\"{text[:60]}\""
        )
        try:
            if self.uses_interactions:
                audio_bytes, mime_type = await self._interactions_with_retry(text)
            else:
                # The preview models sometimes answer in text unless the request is
                # framed as speech.
                tts_prompt = f"Repeat exactly this text as audio: {text}"
                audio_bytes, mime_type = await asyncio.to_thread(self._collect_stream, tts_prompt)
        except Exception as e:
            std_log.error(f"❌ Gemini TTS: API call failed | {type(e).__name__}: {str(e)}")
            raise

        if not audio_bytes or not mime_type:
            raise RuntimeError(f"Gemini TTS ({self.model}) returned no audio")

        total_size = len(audio_bytes)
        std_log.info(
            f"✅ Gemini TTS: Audio received | size={total_size} bytes "
            f"({total_size//1024}KB) mime={mime_type}"
        )

        # Raw L16 PCM needs a RIFF header so clients can play it. Mime casing has
        # varied between model versions ("audio/L16" vs "audio/l16").
        if mime_type.lower().startswith("audio/l16") and not audio_bytes.startswith(b'RIFF'):
            yield _create_wav_header(sample_rate=24000, num_channels=1, sample_width=2,
                                     data_size=total_size)

        # Yield in chunks for streaming over ZMQ/WS
        chunk_size = 32 * 1024
        for i in range(0, total_size, chunk_size):
            yield audio_bytes[i:i + chunk_size]
