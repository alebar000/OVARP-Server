"""
Open Virtual Agent Research Platform (OVARP) — Telemetry Logger

Handles structured logging of all interactions for experimental
reproducibility. Events are recorded in JSONL format with high-precision
timestamps and can be exported to CSV for analysis in SPSS, R, or
similar statistical software.

Now supports session-aware logging: when an experiment session is active,
every log entry includes participant_id and session_id. Event markers
are logged as distinct entries for post-hoc analysis.

Author: Alexander Barquero Elizondo, Ph.D. — UCR, ECCI/CITIC
License: MIT
"""

import atexit
import os
import json
import csv
import queue
import threading
import time
from datetime import datetime, timezone
import structlog
from pathlib import Path
from typing import Optional

from src.core.schemas import BaseCommand


def _decoded_size(encoded: str) -> int:
    """Byte length of a base64 payload, without decoding it."""
    padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
    return len(encoded) * 3 // 4 - padding


def _wav_duration_ms(head: bytes, total_bytes: int) -> Optional[int]:
    """Duration of a PCM WAV from its 44-byte header and total size."""
    if len(head) < 44 or head[:4] != b"RIFF":
        return None
    byte_rate = int.from_bytes(head[28:32], "little")
    if not byte_rate:
        return None
    return round(1000 * (total_bytes - 44) / byte_rate)


class _RecordWriter:
    """Writes the session record on one background thread.

    The event loop only hands over a record (a dict, an already-rendered line, or
    an audio file) and returns; JSON encoding and every disk write happen here,
    in submission order. The file is flushed whenever the queue runs empty, so
    the disk trails the live session by milliseconds.
    """

    def __init__(self, path: Path):
        self.path = path
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="ovarp-record", daemon=True)
        self._thread.start()
        atexit.register(self.flush)

    # -- called from the event loop ---------------------------------------------
    def info(self, **fields) -> None:
        """Queue one record. Same call shape as a structlog logger."""
        self._queue.put(("record", time.time(), fields))

    def info_now(self, **fields) -> None:
        """Encode now, write later: for records whose objects may change after
        the call (marker metadata, questionnaire answers), so the line holds the
        values as they were when the event happened."""
        fields["timestamp"] = (
            datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self.put_line(json.dumps(fields, default=str))

    def put_line(self, line: str) -> None:
        self._queue.put(("line", line))

    def put_file(self, path: Path, chunks: list[bytes]) -> None:
        self._queue.put(("file", path, chunks))

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until everything queued so far is on disk."""
        if not self._thread.is_alive():
            return False
        done = threading.Event()
        self._queue.put(("sync", done))
        return done.wait(timeout)

    # -- writer thread ------------------------------------------------------------
    def _run(self) -> None:
        with open(self.path, "a", encoding="utf-8") as out:
            while True:
                item = self._queue.get()
                try:
                    kind = item[0]
                    if kind == "record":
                        _, ts, fields = item
                        fields["timestamp"] = (
                            datetime.fromtimestamp(ts, tz=timezone.utc)
                            .isoformat().replace("+00:00", "Z")
                        )
                        out.write(json.dumps(fields, default=str) + "\n")
                    elif kind == "line":
                        out.write(item[1] + "\n")
                    elif kind == "file":
                        _, path, chunks = item
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "wb") as f:
                            for chunk in chunks:
                                f.write(chunk)
                    elif kind == "sync":
                        out.flush()
                        item[1].set()
                        continue
                except Exception as e:  # one bad record must not stop the recorder
                    out.write(json.dumps({"event": "record_write_error", "error": repr(e)}) + "\n")
                if self._queue.empty():
                    out.flush()


class _LineSink:
    """structlog logger that hands rendered lines to the record writer."""

    def __init__(self, writer: _RecordWriter):
        self._writer = writer

    def msg(self, message: str) -> None:
        self._writer.put_line(message)

    log = debug = info = warn = warning = err = error = critical = exception = fatal = msg


class TelemetryLogger:
    """
    Handles structured logging of all interactions for experimental reproducibility.
    Logs are saved in JSONL format, and can be exported to CSV.
    Session-aware: injects participant_id and session_id when a session is active.
    """
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path("data/sessions")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"session_{self.session_id}.jsonl"

        # One writer thread owns the session file. Telemetry events reach it as
        # dicts (encoded on that thread); anything else logged through structlog
        # is rendered as before and queued as a line, so both keep their order.
        self._writer = _RecordWriter(self.jsonl_path)
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=lambda *args: _LineSink(self._writer),
        )
        self.file_logger = self._writer

        # Synthesized speech is kept as WAV files beside the record, one per
        # utterance, written by the same thread so files and entries stay in order.
        self._audio_seq = 0

        # Keep a console logger separate so we don't spam JSON to the stdout
        self.console_logger = structlog.get_logger("console")
        self.console_logger.info("Telemetry Session Started", session_id=self.session_id, file=str(self.jsonl_path))

    def _log_now(self, **fields) -> None:
        """Record an event whose payload must be captured as it is at this moment."""
        now = getattr(self.file_logger, "info_now", None)
        (now or self.file_logger.info)(**fields)

    def _get_session_context(self) -> dict:
        """Returns session metadata to inject into every log entry, if a session is active."""
        # Import here to avoid circular imports (session_manager imports nothing from telemetry)
        from src.core.session_manager import session_manager

        if session_manager.is_active and session_manager.session:
            return {
                "experiment_session_id": session_manager.session.session_id,
                "participant_id": session_manager.session.participant_id,
                "session_status": session_manager.session.status,
            }
        return {}

    def log_interaction(self, command: BaseCommand):
        """Append a Command directly to the session logs with an atomic timestamp.

        Audio chunks are recorded by size. The audio itself is saved once per
        utterance as a WAV file (``log_agent_audio``) rather than re-encoded into
        the record as base64 with every chunk.
        """
        ctx = self._get_session_context()
        sub = command.subcommand
        if sub and "audio_base64" in sub:
            sub = dict(sub)
            sub["audio_bytes"] = _decoded_size(sub.pop("audio_base64") or "")
        self.file_logger.info(
            event="interaction",
            host_timestamp=time.time(),
            sender=command.sender,
            target_device=command.target_device,
            target_agent=command.target_agent,
            command_type=command.command_type,
            command=command.command,
            subcommand=sub,
            **ctx
        )

    def log_agent_audio(self, chunks: list[bytes], text: str, target_agent: str) -> Optional[str]:
        """Save one utterance of synthesized speech and record where it is.

        ``chunks`` are the audio exactly as sent to clients (a WAV header followed
        by PCM). The file is written in the background; the record entry, with
        the file name, the spoken text and the duration, is queued now so it keeps
        its place in the timeline. Returns the file's path relative to the
        sessions directory.
        """
        total = sum(len(c) for c in chunks)
        if not total:
            return None
        self._audio_seq += 1
        rel = f"{self.jsonl_path.stem}_audio/{self._audio_seq:05d}.wav"
        head = chunks[0] if len(chunks[0]) >= 44 else b"".join(chunks[:2])[:44]
        self._writer.put_file(self.jsonl_path.parent / rel, list(chunks))
        ctx = self._get_session_context()
        self.file_logger.info(
            event="agent_audio",
            host_timestamp=time.time(),
            target_agent=target_agent,
            text=text,
            audio_file=rel,
            audio_bytes=total,
            duration_ms=_wav_duration_ms(head, total),
            **ctx
        )
        return rel

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until every queued record and audio file is on disk."""
        writer = getattr(self, "_writer", None)
        return writer.flush(timeout) if writer is not None else True

    def log_inbound(self, command: BaseCommand):
        """Record a request a client sent to the server (speech, typed text, or
        operator-injected speech) before the pipeline handles it.

        Outbound traffic is logged in ``log_interaction``; without this entry, a
        typed turn would be visible in the record only through its reply. The
        participant's raw audio is not stored: an ``audio_base64`` payload is
        replaced by its size, and the transcript is recorded when STT returns it.
        """
        ctx = self._get_session_context()
        sub = dict(command.subcommand or {})
        if "audio_base64" in sub:
            sub["audio_bytes"] = _decoded_size(sub.pop("audio_base64") or "")
        self.file_logger.info(
            event="inbound",
            host_timestamp=time.time(),
            sender=command.sender,
            target_device=command.target_device,
            target_agent=command.target_agent,
            command_type=command.command_type,
            command=command.command,
            subcommand=sub,
            **ctx
        )

    def log_marker(self, label: str, metadata: dict = None,
                   marker_id: str = None, category: str = None):
        """Log an event marker as a distinct telemetry entry.

        ``marker_id`` is the marker's stable id in the session, so later
        ``marker_amended`` / ``marker_deleted`` entries can be joined to this one.
        """
        ctx = self._get_session_context()
        self._log_now(
            event="marker",
            host_timestamp=time.time(),
            marker_id=marker_id,
            label=label,
            category=category,
            marker_metadata=metadata,
            **ctx
        )

    def log_marker_amendment(self, marker_id: str, before: dict, after: dict):
        """Record a marker correction without rewriting what was originally logged.

        The session JSONL is append-only, so the first ``marker`` entry stays as
        captured and this adds the diff on top. The pair is what makes an edited
        marker auditable rather than just changed.
        """
        ctx = self._get_session_context()
        self._log_now(
            event="marker_amended",
            host_timestamp=time.time(),
            marker_id=marker_id,
            amended_from=before,
            amended_to=after,
            **ctx
        )

    def log_marker_deleted(self, marker_id: str, label: str):
        """Record that a marker was retracted, keeping the original entry intact."""
        ctx = self._get_session_context()
        self._log_now(
            event="marker_deleted",
            host_timestamp=time.time(),
            marker_id=marker_id,
            label=label,
            **ctx
        )

    def log_survey_response(self, survey_id: str, participant_id: str,
                            answers: dict, score: dict):
        """Record a completed questionnaire alongside the interaction log.

        Keeping responses in the same JSONL as the session means the scores line
        up with the markers and latencies for that participant without a join.
        """
        ctx = self._get_session_context()
        ctx["participant_id"] = participant_id or ctx.get("participant_id", "")
        self._log_now(
            event="survey_response",
            host_timestamp=time.time(),
            survey_id=survey_id,
            survey_answers=answers,
            survey_score=score,
            **ctx
        )

    def log_latency(self, latency: dict):
        """Log pipeline latency metrics."""
        ctx = self._get_session_context()
        self.file_logger.info(
            event="latency",
            host_timestamp=time.time(),
            **latency,
            **ctx
        )

    def log_xr_telemetry(self, device_id: str, frames: list[dict]):
        """Log batched XR telemetry frames (head/hand/gaze tracking)."""
        ctx = self._get_session_context()
        for frame in frames:
            self._log_now(
                event="xr_telemetry",
                host_timestamp=time.time(),
                device_id=device_id,
                **frame,
                **ctx
            )

    def log_session_event(self, event_type: str, details: dict = None):
        """Log session lifecycle events (start, pause, resume, end)."""
        ctx = self._get_session_context()
        self._log_now(
            event="session",
            host_timestamp=time.time(),
            session_event=event_type,
            details=details or {},
            **ctx
        )

    def export_to_csv(self) -> Path:
        """Parses the current session JSONL and flattens it to CSV for statistical analysis software like SPSS/R."""
        csv_path = self.log_dir / f"session_{self.session_id}.csv"
        self.flush()

        try:
            with open(self.jsonl_path, 'r', encoding="utf-8") as f_in, \
                 open(csv_path, 'w', newline='', encoding="utf-8") as f_out:

                writer = csv.writer(f_out)
                # Write header with session columns
                writer.writerow([
                    "iso_time", "host_timestamp", "event_type",
                    "participant_id", "experiment_session_id",
                    "sender", "target_device", "target_agent",
                    "command_type", "command", "subcommand_json",
                    "marker_label", "marker_metadata",
                    "marker_id", "marker_amended_from", "marker_amended_to",
                    "survey_id", "survey_score", "survey_answers",
                    "stt_ms", "llm_ms", "tts_ms", "total_ms"
                ])

                for line in f_in:
                    if not line.strip(): continue
                    data = json.loads(line)
                    event_type = data.get("event", "")

                    writer.writerow([
                        data.get("timestamp", ""),
                        data.get("host_timestamp", ""),
                        event_type,
                        data.get("participant_id", ""),
                        data.get("experiment_session_id", ""),
                        data.get("sender", ""),
                        data.get("target_device", ""),
                        data.get("target_agent", ""),
                        data.get("command_type", ""),
                        data.get("command", ""),
                        json.dumps(data.get("subcommand", {})),
                        data.get("label", ""),
                        json.dumps(data.get("marker_metadata", {})),
                        data.get("marker_id", ""),
                        json.dumps(data.get("amended_from", {})) if data.get("amended_from") else "",
                        json.dumps(data.get("amended_to", {})) if data.get("amended_to") else "",
                        data.get("survey_id", ""),
                        json.dumps(data.get("survey_score", {})) if data.get("survey_score") else "",
                        json.dumps(data.get("survey_answers", {})) if data.get("survey_answers") else "",
                        data.get("stt_ms", ""),
                        data.get("llm_ms", ""),
                        data.get("tts_ms", ""),
                        data.get("total_ms", ""),
                    ])

            self.console_logger.info("Telemetry Exported successfully", csv_file=str(csv_path))
            return csv_path
        except Exception as e:
            self.console_logger.error("Failed to export telemetry to CSV", error=str(e))
            return None

telemetry = TelemetryLogger()
