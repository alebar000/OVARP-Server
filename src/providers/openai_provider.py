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

import os
import json
from typing import AsyncGenerator, Dict, Any, Optional
from openai import AsyncOpenAI
import logging
from structlog import get_logger

from src.providers.base import BaseSTTProvider, BaseLLMProvider, BaseTTSProvider
from src.core.config import config_manager

logger = get_logger()
std_log = logging.getLogger("OVARP.openai")

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
        self.model = model_name or os.getenv("OVARP_OPENAI_STT_MODEL", "whisper-1")

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
                           system_prompt: Optional[str], history: Optional[list]
                           ) -> AsyncGenerator[str, None]:
    """Yield the reply's text as the endpoint generates it. Errors propagate, so
    the orchestrator reports a ``pipeline_error`` instead of ending in silence."""
    stream = await client.chat.completions.create(
        model=model,
        messages=chat_messages(prompt, system_prompt, history, plain_text=True),
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue  # e.g. a trailing usage-only chunk
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def classify_actions(client: AsyncOpenAI, model: str, spoken_text: str,
                           system_prompt: Optional[str], tool_choice: Any = None
                           ) -> Dict[str, Any]:
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
        self.model = model_name or os.getenv("OVARP_OPENAI_LLM_MODEL", "gpt-4o")

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
            stream=True
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
            tool_choice="required"  # Force the model to always call update_agent_state
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
                                                system_prompt, history):
                yield delta
        except Exception as e:
            std_log.error(f"❌ OpenAI: Streaming text call FAILED | {type(e).__name__}: {e}")
            raise

    async def extract_actions(self, spoken_text: str,
                              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Gestures and emotion for a reply that was already streamed."""
        try:
            return await classify_actions(self.client, self.model, spoken_text, system_prompt)
        except Exception as e:
            std_log.warning(f"⚠️ OpenAI: extract_actions call failed | {type(e).__name__}: {e}")
            return {}

class OpenAITTSProvider(BaseTTSProvider):
    """OpenAI TTS streaming provider."""
    def __init__(self, model_name: str = None, voice: str = "alloy"):
        self.model = model_name or os.getenv("OVARP_OPENAI_TTS_MODEL", "tts-1")
        self.voice = voice

    @property
    def client(self) -> AsyncOpenAI:
        return OpenAIClientSingleton.get_client()

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        std_log.info(f"🔊 TTS: Starting synthesis | model={self.model} voice={self.voice} text=\"{text[:60]}\"")
        audio_response = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav"  # WAV allows easier playback in Unity and browsers
        )
        
        # Read the full audio response (.content is sync-safe on the already-resolved response)
        # Note: iter_bytes() returns a SYNC iterator on AsyncOpenAI responses,
        # so 'async for' silently yields nothing. We read .content instead.
        audio_bytes = audio_response.content
        total_size = len(audio_bytes)
        std_log.info(f"✅ TTS: Audio received | size={total_size} bytes ({total_size//1024}KB)")
        
        # Yield in chunks for streaming over ZMQ/WS
        chunk_size = 32 * 1024  # 32KB chunks
        chunk_count = 0
        for i in range(0, total_size, chunk_size):
            chunk_count += 1
            yield audio_bytes[i:i + chunk_size]
        std_log.info(f"📤 TTS: Yielded {chunk_count} chunks to transport")
