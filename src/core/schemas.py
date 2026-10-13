"""
Open Virtual Agent Research Platform (OVARP) — Command Schemas

Defines the ``BaseCommand`` Pydantic model that is the universal message
structure flowing through ZMQ and WebSocket transports. Commands are
validated dynamically against the current experiment configuration loaded
from ``config.yaml``.

Author: Alexander Barquero Elizondo, Ph.D. — UCR, ECCI/CITIC
License: MIT
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, Optional, Literal
import base64
import json

from src.core.config import config_manager

try:  # SIMD base64, about 25x faster on 32 KB audio chunks; stdlib otherwise
    from pybase64 import b64encode_as_string as _b64_text
except ImportError:  # pragma: no cover - exercised only without the optional wheel
    def _b64_text(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")


class Base64Text(str):
    """Base64 produced by the server itself.

    The base64 alphabet needs no escaping in JSON, so a command whose only
    subcommand field is one of these can be serialized by splicing the string in
    instead of having the JSON encoder scan tens of kilobytes per audio chunk.
    Text that merely looks like base64 (e.g. from a client) is a plain ``str``
    and takes the normal path.
    """

    __slots__ = ()


def encode_audio(data: bytes) -> Base64Text:
    """Encode an audio chunk for a ``tts_chunk`` command."""
    return Base64Text(_b64_text(data))

class BaseCommand(BaseModel):
    """
    Universal Command structure that flows through ZMQ and WebSockets.
    Validated dynamically against the current Experiment Configuration.
    """
    sender: str = Field(description="Entity sending the command (e.g., 'server', 'agent_alpha', 'web_panel_01')")
    target_device: str = Field(description="Target device ID or 'all'")
    target_agent: str = Field(description="Target agent ID or 'all'")
    command_type: Literal["audio", "message", "action", "scene_change", "system"]
    command: str = Field(description="Main instruction or payload")
    subcommand: Optional[Dict[str, Any]] = Field(default=None, description="Additional structured parameters")

    @field_validator('target_device')
    def validate_target_device(cls, v):
        valid = config_manager.get_valid_devices()
        if v not in valid:
            raise ValueError(f"Invalid target_device '{v}'. Must be one of {valid}")
        return v
        
    @field_validator('target_agent')
    def validate_target_agent(cls, v):
        valid = config_manager.get_valid_agents()
        if v not in valid:
            raise ValueError(f"Invalid target_agent '{v}'. Must be one of {valid}")
        return v
        
    @field_validator('subcommand')
    def validate_subcommand(cls, v):
        if v is None:
            return v
        
        # Only validate action/state subcommands against our custom dicts for now
        config = config_manager.config
        if not config or not config.custom_commands:
            return v

        for key, val in v.items():
            if key in config.custom_commands:
                valid_vals = config.custom_commands[key].values
                if val not in valid_vals:
                    raise ValueError(f"Invalid value '{val}' for command category '{key}'. Must be in {valid_vals}")
        return v

    @model_validator(mode="after")
    def validate_state_categories(self):
        """An ``execute_state`` may only name declared categories.

        Values are checked per category above; this closes the other half, so a
        misspelled category key is rejected instead of passing through unseen.
        """
        if self.command != "execute_state" or not self.subcommand:
            return self
        config = config_manager.config
        if not config or not config.custom_commands:
            return self
        unknown = [k for k in self.subcommand if k not in config.custom_commands]
        if unknown:
            raise ValueError(
                f"Unknown command category {unknown} in execute_state. "
                f"Declared categories: {list(config.custom_commands)}"
            )
        return self

    def to_json(self) -> str:
        sub = self.subcommand
        if sub is not None and len(sub) == 1:
            audio = sub.get("audio_base64")
            if type(audio) is Base64Text:
                # Same bytes model_dump_json would produce (pinned by a test).
                head = self.model_dump_json(exclude={"subcommand"})
                return f'{head[:-1]},"subcommand":{{"audio_base64":"{audio}"}}}}'
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "BaseCommand":
        return cls(**json.loads(json_str))
