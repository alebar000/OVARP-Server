"""
OVARP latency benchmark: full-reply synthesis vs. sentence-level TTS.

Runs the real DialogOrchestrator in-process (no browser, no headset) and
timestamps every message it emits at the point where it leaves the server
(the router's outbound dispatch). That isolates server-side pipeline latency
from client network and playback, which is what the paper reports.

Two scheduling modes are compared on the same provider pair:

  full      -- the LLM produces the whole reply (and its gesture/emotion tool
               call) in one request, then the whole reply is synthesized.
               This is how OVARP behaved before OPA-303/OPA-335, and how it
               still behaves for LLM providers without ``stream_reply``.
  sentence  -- the LLM streams text; each completed sentence goes to TTS at
               once (up to 3 in flight) and its audio is sent as soon as it is
               ready, while gesture/emotion classification runs off the audio
               path (``extract_actions``). Only for LLMs with ``stream_reply``
               (currently Gemini).

Metrics per turn (all ms, measured from the moment the turn is submitted):
  ttft_ms     time to first text chunk (sentence mode only)
  reply_ms    time to the complete ``llm_reply`` broadcast
  ttfa_ms     time to first audio  (first ``tts_chunk`` leaving the server)
  done_ms     time to the last ``tts_complete``
  plus the server's own latency record (stt/llm/tts/total, tts_first_chunk).

Usage
-----
  # 1) dry run, no keys, validates the harness (simulated providers):
  python scripts/bench_latency.py --mock --reps 3

  # 2) platform overhead only (zero-latency providers):
  python scripts/bench_latency.py --mock --mock-zero --reps 20

  # 3) the real benchmark (needs GEMINI_API_KEY, and OPENAI_API_KEY for openai):
  python scripts/bench_latency.py --pairs gemini:gemini,gemini:openai,openai:openai \
      --reps 10 --gap 2.0 --out bench_results

Each run writes <out>/<run_id>/trials.csv, summary.csv, summary.md and env.json.
Re-summarize an existing run with:  python scripts/bench_latency.py --summarize <dir>

Author: Alexander Barquero Elizondo, Ph.D. -- UCR, ECCI/CITIC
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config.yaml, profiles/ and data/ are resolved relative to the repo root

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.core.config import config_manager  # noqa: E402
from src.core.router import router  # noqa: E402
from src.core.orchestrator import DialogOrchestrator  # noqa: E402
from src.providers.base import BaseLLMProvider, BaseTTSProvider, BaseSTTProvider  # noqa: E402
from src.transport.base import BaseTransport  # noqa: E402

# --------------------------------------------------------------------------
# Prompt set. Three length classes, four prompts each, so the effect of
# sentence-level scheduling can be read as a function of reply length.
# The system prompt below asks for a reply length explicitly; the actual
# length is recorded per trial (reply_chars, n_sentences) and is what the
# analysis should condition on.
# --------------------------------------------------------------------------
PROMPTS: list[dict[str, str]] = [
    {"id": "s1", "cls": "short", "text": "Hi! How are you today?"},
    {"id": "s2", "cls": "short", "text": "What is your name?"},
    {"id": "s3", "cls": "short", "text": "Can you hear me well?"},
    {"id": "s4", "cls": "short", "text": "Say good morning to me."},
    {"id": "m1", "cls": "medium", "text": "In two or three sentences, what do you like about rainy days?"},
    {"id": "m2", "cls": "medium", "text": "In two or three sentences, recommend a book for a long trip and say why."},
    {"id": "m3", "cls": "medium", "text": "In two or three sentences, how would you calm someone before an exam?"},
    {"id": "m4", "cls": "medium", "text": "In two or three sentences, describe your favourite place in a city."},
    {"id": "l1", "cls": "long", "text": "In about six sentences, explain how to prepare for a job interview."},
    {"id": "l2", "cls": "long", "text": "In about six sentences, tell me a short story about a lost dog that finds its way home."},
    {"id": "l3", "cls": "long", "text": "In about six sentences, explain why sleep matters for memory."},
    {"id": "l4", "cls": "long", "text": "In about six sentences, describe a day at a busy market."},
]

BENCH_SYSTEM_PROMPT = (
    "You are a friendly embodied virtual agent in a research study. "
    "Answer conversationally, in plain spoken English, with no lists, markdown or emoji. "
    "Follow any length the user asks for; otherwise answer in one sentence."
)

TARGET_DEVICE = "web_01"
TARGET_AGENT = "agent_alpha"


# --------------------------------------------------------------------------
# Capture transport: records when each outbound message leaves the server.
# --------------------------------------------------------------------------
class CaptureTransport(BaseTransport):
    def __init__(self):
        super().__init__()
        self.events: list[tuple[float, str, dict]] = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, target_device: str, topic: str, message: str):
        t = time.perf_counter()
        try:
            payload = json.loads(message)
        except Exception:
            payload = {}
        self.events.append((t, payload.get("command", "?"), payload.get("subcommand") or {}))


class NoStreamLLM:
    """Wraps an LLM provider and hides ``stream_reply`` so the orchestrator takes the
    full-reply path. Everything else is delegated untouched."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name in ("stream_reply", "extract_actions"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


# --------------------------------------------------------------------------
# Mock providers (for --mock): a simple, documented latency model.
# --------------------------------------------------------------------------
@dataclass
class MockModel:
    llm_first_token_ms: float = 450.0     # time to first token
    llm_tokens_per_s: float = 60.0        # generation speed (1 token ~ 4 chars)
    actions_ms: float = 700.0             # separate gesture/emotion classification call
    tts_base_ms: float = 600.0            # fixed TTS request overhead
    tts_ms_per_char: float = 9.0          # synthesis time per character
    jitter: float = 0.15                  # +/- relative noise

    def j(self, x: float) -> float:
        return max(0.0, x * (1 + random.uniform(-self.jitter, self.jitter)))


MOCK_REPLIES = {
    "short": "I'm doing well, thank you for asking.",
    "medium": ("Rainy days make everything feel slower and calmer. "
               "I like the sound on the windows. It is a good excuse to read."),
    "long": ("Start by learning about the company and the role. "
             "Then write down three stories that show your strengths. "
             "Practice answering common questions out loud. "
             "Prepare two questions to ask the interviewer. "
             "Plan your route so you arrive a little early. "
             "Finally, get some sleep the night before."),
}


class MockLLM(BaseLLMProvider):
    model = "mock-llm"

    def __init__(self, m: MockModel, zero: bool):
        self.m, self.zero = m, zero
        self._cls = "short"

    def set_class(self, cls: str):
        self._cls = cls

    async def _sleep(self, ms: float):
        if not self.zero and ms > 0:
            await asyncio.sleep(self.m.j(ms) / 1000)

    async def generate_response(self, prompt, system_prompt=None):  # pragma: no cover
        yield MOCK_REPLIES[self._cls]

    async def generate_response_with_actions(self, prompt, system_prompt=None, history=None):
        text = MOCK_REPLIES[self._cls]
        await self._sleep(self.m.llm_first_token_ms + 1000 * (len(text) / 4) / self.m.llm_tokens_per_s)
        return text, {"emotions": "happy", "actions": "nod"}

    async def stream_reply(self, prompt, system_prompt=None, history=None):
        text = MOCK_REPLIES[self._cls]
        await self._sleep(self.m.llm_first_token_ms)
        words = text.split(" ")
        for i, w in enumerate(words):
            await self._sleep(1000 * (len(w) + 1) / 4 / self.m.llm_tokens_per_s)
            yield w + (" " if i < len(words) - 1 else "")

    async def extract_actions(self, spoken_text, system_prompt=None):
        await self._sleep(self.m.actions_ms)
        return {"emotions": "happy", "actions": "nod"}


class MockTTS(BaseTTSProvider):
    model = "mock-tts"

    def __init__(self, m: MockModel, zero: bool):
        self.m, self.zero = m, zero
        self.voice = "mock"

    async def synthesize_stream(self, text: str):
        if not self.zero:
            await asyncio.sleep(self.m.j(self.m.tts_base_ms + self.m.tts_ms_per_char * len(text)) / 1000)
        n = 24000 * 2 * max(1, len(text)) // 15  # ~15 chars/s of speech, 24 kHz 16-bit mono
        hdr = b"RIFF" + (36 + n).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little") \
            + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (24000).to_bytes(4, "little") \
            + (48000).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little") \
            + b"data" + n.to_bytes(4, "little")
        yield hdr
        for i in range(0, n, 32 * 1024):
            yield b"\x00" * min(32 * 1024, n - i)


class MockSTT(BaseSTTProvider):
    async def transcribe(self, audio_data: bytes) -> str:
        return ""


# --------------------------------------------------------------------------
# Trial bookkeeping
# --------------------------------------------------------------------------
@dataclass
class Trial:
    run_id: str
    trial: int
    llm: str
    tts: str
    llm_model: str
    tts_model: str
    mode: str
    prompt_id: str
    length_class: str
    rep: int
    order_in_block: int
    started_unix: float = 0.0
    ttft_ms: Optional[float] = None
    reply_ms: Optional[float] = None
    ttfa_ms: Optional[float] = None
    done_ms: Optional[float] = None
    actions_ms: Optional[float] = None
    reply_chars: int = 0
    n_sentences: int = 0
    n_utterances: int = 0
    n_tts_chunks: int = 0
    audio_bytes: int = 0
    audio_s: Optional[float] = None
    speech_end_ms: Optional[float] = None   # modeled: when a client playing in order finishes speaking
    stall_ms: Optional[float] = None        # modeled: silence between sentences waiting for audio
    n_stalls: int = 0
    srv_stt_ms: Optional[float] = None
    srv_llm_ms: Optional[float] = None
    srv_llm_first_chunk_ms: Optional[float] = None
    srv_tts_ms: Optional[float] = None
    srv_tts_first_chunk_ms: Optional[float] = None
    srv_total_ms: Optional[float] = None
    error: str = ""
    reply_text: str = ""


def _wav_seconds(first_chunk_b64: str, total_bytes: int) -> Optional[float]:
    import base64
    try:
        head = base64.b64decode(first_chunk_b64)[:44]
        if head[:4] != b"RIFF":
            return None
        byte_rate = int.from_bytes(head[28:32], "little")
        return (total_bytes - 44) / byte_rate if byte_rate else None
    except Exception:
        return None


def _count_sentences(text: str) -> int:
    from src.core.orchestrator import _split_ready_sentences
    ready, rest = _split_ready_sentences(text + " ")
    return len(ready) + (1 if rest.strip() else 0)


def build_orchestrator(pair: tuple[str, str], mock: bool, zero: bool, mm: MockModel):
    llm_id, tts_id = pair
    if mock:
        llms = {llm_id: MockLLM(mm, zero)}
        ttss = {tts_id: MockTTS(mm, zero)}
        stts = {"mock": MockSTT()}
    else:
        llms, ttss, stts = {}, {}, {}
        if llm_id == "gemini" or tts_id == "gemini":
            from src.providers.gemini_provider import GeminiLLMProvider, GeminiTTSProvider, GeminiSTTProvider
            llms["gemini"] = GeminiLLMProvider()
            ttss["gemini"] = GeminiTTSProvider()
            stts["gemini"] = GeminiSTTProvider()
        if llm_id == "openai" or tts_id == "openai":
            from src.providers.openai_provider import OpenAILLMProvider, OpenAITTSProvider, OpenAISTTProvider
            llms["openai"] = OpenAILLMProvider()
            ttss["openai"] = OpenAITTSProvider()
            stts["openai"] = OpenAISTTProvider()
    orch = DialogOrchestrator(
        stt_providers=stts, llm_providers=llms, tts_providers=ttss,
        default_llm=llm_id, default_tts=tts_id, default_stt=next(iter(stts)),
    )
    orch.set_system_prompt(BENCH_SYSTEM_PROMPT)
    return orch


def supports_sentence(orch: DialogOrchestrator, llm_id: str) -> bool:
    return hasattr(orch.llm_providers[llm_id], "stream_reply")


async def run_turn(orch: DialogOrchestrator, cap: CaptureTransport, tr: Trial, text: str,
                   llm_obj, timeout_s: float) -> Trial:
    orch.llm_providers[tr.llm] = NoStreamLLM(llm_obj) if tr.mode == "full" else llm_obj
    orch.clear_history()
    orch.clear_history(TARGET_AGENT)
    cap.events.clear()
    tr.started_unix = time.time()
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(orch.process_text_interaction(text, TARGET_DEVICE, TARGET_AGENT), timeout_s)
    except asyncio.TimeoutError:
        tr.error = f"timeout>{timeout_s}s"
    ms = lambda t: round((t - t0) * 1000, 1)  # noqa: E731
    first_audio_b64 = None
    utter_first = None
    utters: list[list] = []   # [arrival_ms_of_complete_audio, bytes, first_chunk_b64]
    cur_bytes = 0
    for t, cmd, sub in cap.events:
        if cmd == "llm_reply_chunk" and tr.ttft_ms is None:
            tr.ttft_ms = ms(t)
        elif cmd == "llm_reply":
            tr.reply_ms = ms(t)
            tr.reply_text = sub.get("text", "")
        elif cmd == "tts_chunk":
            b = sub.get("audio_base64", "")
            if tr.ttfa_ms is None:
                tr.ttfa_ms = ms(t)
            if utter_first is None:
                utter_first = b
            tr.n_tts_chunks += 1
            tr.audio_bytes += len(b) * 3 // 4
            cur_bytes += len(b) * 3 // 4
        elif cmd == "tts_complete":
            tr.done_ms = ms(t)
            tr.n_utterances += 1
            if utter_first is not None:
                first_audio_b64 = first_audio_b64 or utter_first
                utters.append([ms(t), cur_bytes, utter_first])
            utter_first = None
            cur_bytes = 0
        elif cmd == "execute_state" and tr.actions_ms is None:
            tr.actions_ms = ms(t)
        elif cmd == "latency":
            tr.srv_stt_ms = sub.get("stt_ms")
            tr.srv_llm_ms = sub.get("llm_ms")
            tr.srv_llm_first_chunk_ms = sub.get("llm_first_chunk_ms") or None
            tr.srv_tts_ms = sub.get("tts_ms")
            tr.srv_tts_first_chunk_ms = sub.get("tts_first_chunk_ms") or None
            tr.srv_total_ms = sub.get("total_ms")
        elif cmd == "pipeline_error":
            tr.error = (tr.error + "; " if tr.error else "") + f"{sub.get('stage')}:{sub.get('error')}:{str(sub.get('detail'))[:120]}"
    if first_audio_b64 and tr.audio_bytes:
        # duration from the WAV byte rate; each utterance carries its own 44-byte header
        try:
            import base64
            head = base64.b64decode(first_audio_b64)[:44]
            br = int.from_bytes(head[28:32], "little")
            if head[:4] == b"RIFF" and br:
                tr.audio_s = round((tr.audio_bytes - 44 * tr.n_utterances) / br, 2)
        except Exception:
            pass
    # Model in-order playback: utterance k starts when its audio has arrived and the
    # previous one has finished; any wait in between is a stall the listener hears.
    try:
        import base64
        play_end, stall, n_st = None, 0.0, 0
        for arr, nbytes, b64 in utters:
            head = base64.b64decode(b64)[:44]
            br = int.from_bytes(head[28:32], "little") if head[:4] == b"RIFF" else 0
            dur = 1000 * (nbytes - 44) / br if br else 0.0
            if play_end is None:
                start = arr
            else:
                start = max(arr, play_end)
                if arr > play_end + 1:  # >1 ms counts as an audible gap
                    stall += arr - play_end
                    n_st += 1
            play_end = start + dur
        if play_end is not None:
            tr.speech_end_ms, tr.stall_ms, tr.n_stalls = round(play_end, 1), round(stall, 1), n_st
    except Exception:
        pass
    tr.reply_chars = len(tr.reply_text)
    tr.n_sentences = _count_sentences(tr.reply_text) if tr.reply_text else 0
    if tr.ttfa_ms is None and not tr.error:
        tr.error = "no_audio"
    return tr


async def run(args) -> Path:
    random.seed(args.seed)
    config_manager.load_config()
    cap = CaptureTransport()
    router.add_transport(cap)

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    modes = args.modes.split(",")
    prompts = [p for p in PROMPTS if not args.classes or p["cls"] in args.classes.split(",")]
    mm = MockModel()
    run_id = time.strftime("%Y%m%d_%H%M%S") + ("_mock0" if args.mock_zero else "_mock" if args.mock else "")
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)

    env = {
        "run_id": run_id, "argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "pairs": pairs, "modes": modes, "reps": args.reps, "gap_s": args.gap, "seed": args.seed,
        "mock": args.mock, "mock_zero": args.mock_zero, "mock_model": asdict(mm) if args.mock else None,
        "system_prompt": BENCH_SYSTEM_PROMPT, "prompts": prompts,
        "gemini_llm_model": os.getenv("OVARP_GEMINI_LLM_MODEL", "gemini-flash-latest"),
        "gemini_tts_model": os.getenv("OVARP_GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
        "notes": args.notes,
    }
    trials: list[Trial] = []
    n = 0
    for pair in pairs:
        orch = build_orchestrator(pair, args.mock, args.mock_zero, mm)
        llm_obj = orch.llm_providers[pair[0]]
        pair_modes = [m for m in modes if m != "sentence" or supports_sentence(orch, pair[0])]
        if len(pair_modes) < len(modes):
            print(f"[{pair[0]}:{pair[1]}] LLM has no stream_reply -> sentence mode skipped")
        env.setdefault("models", {})[f"{pair[0]}:{pair[1]}"] = {
            "llm_model": getattr(llm_obj, "model", "?"),
            "tts_model": getattr(orch.tts_providers[pair[1]], "model", "?"),
            "tts_voice": getattr(orch.tts_providers[pair[1]], "voice", "?"),
        }
        # warm-up turns (not recorded): connection setup, DNS, TLS, model cold start
        for m in pair_modes:
            for _ in range(args.warmup):
                if args.mock:
                    llm_obj.set_class("short")
                w = Trial(run_id, -1, pair[0], pair[1], "", "", m, "warmup", "short", 0, 0)
                await run_turn(orch, cap, w, PROMPTS[0]["text"], llm_obj, args.timeout)
        for rep in range(args.reps):
            order = prompts[:]
            random.shuffle(order)
            for p in order:
                # Counterbalance: which mode goes first alternates across prompts within a
                # repetition (half start with each) and, for a given prompt, across repetitions.
                pidx = next(i for i, q in enumerate(prompts) if q["id"] == p["id"])
                block = pair_modes if (rep + pidx) % 2 == 0 else list(reversed(pair_modes))
                for k, m in enumerate(block):
                    if args.mock:
                        llm_obj.set_class(p["cls"])
                    tr = Trial(run_id, n, pair[0], pair[1], getattr(llm_obj, "model", "?"),
                               getattr(orch.tts_providers[pair[1]], "model", "?"),
                               m, p["id"], p["cls"], rep, k)
                    tr = await run_turn(orch, cap, tr, p["text"], llm_obj, args.timeout)
                    trials.append(tr)
                    n += 1
                    print(f"{n:4d} {pair[0]}:{pair[1]} {m:8s} {p['id']} rep{rep}  "
                          f"ttfa={tr.ttfa_ms}  done={tr.done_ms}  chars={tr.reply_chars} {tr.error}")
                    if args.gap and not args.mock:
                        await asyncio.sleep(args.gap)
        orch.llm_providers[pair[0]] = llm_obj

    with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trials[0]).keys()))
        w.writeheader()
        for t in trials:
            w.writerow(asdict(t))
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    summarize(out)
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _boot_ci_median(xs: list[float], b: int = 5000, seed: int = 7) -> tuple[float, float]:
    rnd = random.Random(seed)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    meds = sorted(statistics.median(rnd.choices(xs, k=len(xs))) for _ in range(b))
    return meds[int(0.025 * b)], meds[int(0.975 * b)]


def _wilcoxon_p(d: list[float]) -> Optional[float]:
    try:
        from scipy.stats import wilcoxon
        d = [x for x in d if x != 0]
        return float(wilcoxon(d).pvalue) if len(d) >= 6 else None
    except Exception:
        return None


def summarize(out: Path) -> None:
    rows = list(csv.DictReader(open(out / "trials.csv", encoding="utf-8")))
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "None") else None  # noqa: E731
    ok = [r for r in rows if not r["error"]]
    groups: dict[tuple, list] = {}
    for r in ok:
        for cls in (r["length_class"], "all"):
            groups.setdefault((r["llm"], r["tts"], r["mode"], cls), []).append(r)

    lines = ["# OVARP latency benchmark", "",
             f"Run: `{out.name}`  |  trials: {len(rows)}  |  failed: {len(rows) - len(ok)}", "",
             "Time-to-first-audio (TTFA) and time-to-last-audio (done), ms, measured at server egress.", "",
             "speech_end = modeled end of speech for a client playing utterances in order; stalls = audible gaps.", "",
             "| LLM:TTS | mode | class | n | TTFA median [IQR] | TTFA p90 | done median [IQR] | speech end median | turns with stalls | stall median (stalled turns) | reply chars (median) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = []
    for (llm, tts, mode, cls), rs in sorted(groups.items()):
        ttfa = [num(r, "ttfa_ms") for r in rs if num(r, "ttfa_ms") is not None]
        done = [num(r, "done_ms") for r in rs if num(r, "done_ms") is not None]
        chars = [num(r, "reply_chars") for r in rs]
        send = [num(r, "speech_end_ms") for r in rs if num(r, "speech_end_ms") is not None]
        stalled = [num(r, "stall_ms") for r in rs if num(r, "stall_ms")]
        if not ttfa:
            continue
        rec = dict(llm=llm, tts=tts, mode=mode, length_class=cls, n=len(ttfa),
                   ttfa_median=round(statistics.median(ttfa)), ttfa_q1=round(_q(ttfa, .25)),
                   ttfa_q3=round(_q(ttfa, .75)), ttfa_p90=round(_q(ttfa, .9)),
                   done_median=round(statistics.median(done)) if done else None,
                   done_q1=round(_q(done, .25)) if done else None, done_q3=round(_q(done, .75)) if done else None,
                   speech_end_median=round(statistics.median(send)) if send else None,
                   turns_with_stalls=len(stalled),
                   stall_median=round(statistics.median(stalled)) if stalled else 0,
                   chars_median=round(statistics.median(chars)))
        summ.append(rec)
        lines.append(f"| {llm}:{tts} | {mode} | {cls} | {rec['n']} | {rec['ttfa_median']} "
                     f"[{rec['ttfa_q1']}-{rec['ttfa_q3']}] | {rec['ttfa_p90']} | {rec['done_median']} "
                     f"[{rec['done_q1']}-{rec['done_q3']}] | {rec['speech_end_median']} | {rec['turns_with_stalls']}/{rec['n']} | "
                     f"{rec['stall_median']} | {rec['chars_median']} |")

    # Paired comparison: same pair, prompt and rep, sentence minus full.
    lines += ["", "## Paired effect of sentence-level scheduling (sentence - full)", "",
              "| LLM:TTS | class | pairs | TTFA diff median | 95% CI (bootstrap) | Wilcoxon p | TTFA ratio (median) | done diff median |",
              "|---|---|---|---|---|---|---|---|"]
    idx = {(r["llm"], r["tts"], r["mode"], r["prompt_id"], r["rep"]): r for r in ok}
    pairs_out = []
    for llm, tts in sorted({(r["llm"], r["tts"]) for r in ok}):
        for cls in ("short", "medium", "long", "all"):
            d_ttfa, d_done, ratio = [], [], []
            for (l2, t2, m, pid, rep), r in idx.items():
                if (l2, t2, m) != (llm, tts, "sentence") or (cls != "all" and r["length_class"] != cls):
                    continue
                f = idx.get((llm, tts, "full", pid, rep))
                if not f or num(f, "ttfa_ms") is None or num(r, "ttfa_ms") is None:
                    continue
                d_ttfa.append(num(r, "ttfa_ms") - num(f, "ttfa_ms"))
                ratio.append(num(r, "ttfa_ms") / num(f, "ttfa_ms"))
                if num(f, "done_ms") is not None and num(r, "done_ms") is not None:
                    d_done.append(num(r, "done_ms") - num(f, "done_ms"))
            if not d_ttfa:
                continue
            lo, hi = _boot_ci_median(d_ttfa)
            p = _wilcoxon_p(d_ttfa)
            rec = dict(llm=llm, tts=tts, length_class=cls, pairs=len(d_ttfa),
                       ttfa_diff_median=round(statistics.median(d_ttfa)), ci_lo=round(lo), ci_hi=round(hi),
                       wilcoxon_p=p, ttfa_ratio_median=round(statistics.median(ratio), 3),
                       done_diff_median=round(statistics.median(d_done)) if d_done else None)
            pairs_out.append(rec)
            ptxt = "n/a" if p is None else (f"{p:.2g}")
            lines.append(f"| {llm}:{tts} | {cls} | {len(d_ttfa)} | {rec['ttfa_diff_median']} | "
                         f"[{rec['ci_lo']}, {rec['ci_hi']}] | {ptxt} | {rec['ttfa_ratio_median']} | {rec['done_diff_median']} |")
    if rows and len(rows) > len(ok):
        lines += ["", "## Failed trials", ""] + [f"- {r['trial']} {r['llm']}:{r['tts']} {r['mode']} {r['prompt_id']}: {r['error']}"
                                                  for r in rows if r["error"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, data in (("summary.csv", summ), ("paired.csv", pairs_out)):
        if data:
            with open(out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    print("\n".join(lines))
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="gemini:gemini", help="comma list of LLM:TTS, e.g. gemini:gemini,gemini:openai")
    ap.add_argument("--modes", default="full,sentence")
    ap.add_argument("--classes", default="", help="subset of short,medium,long")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between turns (rate limits)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="bench_results")
    ap.add_argument("--notes", default="", help="free text stored in env.json (network, machine, location)")
    ap.add_argument("--mock", action="store_true", help="simulated providers, no API keys")
    ap.add_argument("--mock-zero", action="store_true", help="with --mock: zero provider latency (platform overhead)")
    ap.add_argument("--summarize", default="", help="re-run the analysis on an existing run directory")
    args = ap.parse_args()
    if args.summarize:
        summarize(Path(args.summarize))
        return
    if args.mock_zero:
        args.mock = True
    if args.mock:
        args.pairs = ",".join("mock:mock" for _ in args.pairs.split(",")[:1])
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
