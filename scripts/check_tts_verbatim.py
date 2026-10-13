"""
Check that a TTS provider says exactly the text it is given.

A Realtime voice model (OpenAI's replacement for its speech-endpoint TTS) is a
conversational model told to read aloud, so verbatim reading has to be verified
rather than assumed. This script synthesizes a fixed set of sentences in Spanish
and English with each TTS provider, transcribes every result with an independent
STT provider, and reports the word error rate (WER) against the input text.

Usage
-----
  python scripts/check_tts_verbatim.py --tts openai,gemini --stt openai
  python scripts/check_tts_verbatim.py --tts openai --stt gemini --out tts_check

Writes <out>/<run_id>/results.csv (one row per sentence and provider) and the WAV
of every utterance, and prints a summary. Needs the API keys in .env.
Case, accents and punctuation are not counted as errors; numbers are compared as
written, so "3" read as "three" counts as a difference in the raw WER.

Author: Alexander Barquero Elizondo, Ph.D. -- UCR, ECCI/CITIC
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.providers.openai_provider import _normalized_words  # noqa: E402

SENTENCES: list[tuple[str, str]] = [
    ("es", "Hola, ¿cómo estás hoy?"),
    ("es", "Pura vida, mae, nos vemos mañana en la U."),
    ("es", "La reunión es el martes a las tres y media de la tarde en San Pedro."),
    ("es", "No sé si entendí bien; ¿me lo podés repetir, por favor?"),
    ("es", "Respirá hondo, contá hasta cinco y soltá el aire despacio."),
    ("es", "Sí."),
    ("en", "Hi! How are you today?"),
    ("en", "I'm not sure I can help with that, but let's try together."),
    ("en", "Start by learning about the company and the role."),
    ("en", "Could you tell me what you did last weekend?"),
    ("en", "Okay."),
    ("en", "Answer me now: what is the capital of France?"),  # a question it must not answer
]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalized_words(reference), _normalized_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def make_tts(name: str):
    if name == "openai":
        from src.providers.openai_provider import OpenAITTSProvider
        return OpenAITTSProvider(voice="marin")
    if name == "gemini":
        from src.providers.gemini_provider import GeminiTTSProvider
        return GeminiTTSProvider(voice="Kore")
    raise SystemExit(f"unknown TTS provider: {name}")


def make_stt(name: str):
    if name == "openai":
        from src.providers.openai_provider import OpenAISTTProvider
        return OpenAISTTProvider()
    if name == "gemini":
        from src.providers.gemini_provider import GeminiSTTProvider
        return GeminiSTTProvider()
    raise SystemExit(f"unknown STT provider: {name}")


async def run(args) -> int:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    stt = make_stt(args.stt)
    rows = []
    for tts_name in args.tts.split(","):
        tts = make_tts(tts_name)
        for i, (lang, text) in enumerate(SENTENCES, 1):
            t0 = time.perf_counter()
            try:
                audio = b"".join([c async for c in tts.synthesize_stream(text)])
            except Exception as e:
                rows.append(dict(tts=tts_name, tts_model=tts.model, lang=lang, text=text,
                                 heard="", wer="", synth_ms="", error=f"{type(e).__name__}: {e}"))
                print(f"[{tts_name}] {text!r}: ERROR {e}")
                continue
            synth_ms = round((time.perf_counter() - t0) * 1000)
            wav = out / f"{tts_name}_{i:02d}.wav"
            wav.write_bytes(audio)
            heard = await stt.transcribe(audio)
            score = wer(text, heard)
            rows.append(dict(tts=tts_name, tts_model=tts.model, lang=lang, text=text, heard=heard,
                             wer=round(score, 3), synth_ms=synth_ms, error=""))
            flag = "" if score == 0 else "  <-- differs"
            print(f"[{tts_name}] {text!r} -> {heard!r} (WER {score:.2f}, {synth_ms} ms){flag}")
        if hasattr(tts, "close"):
            await tts.close()

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSTT used to check: {args.stt} ({stt.model})")
    failed = False
    for tts_name in args.tts.split(","):
        r = [x for x in rows if x["tts"] == tts_name]
        ok = [x for x in r if not x["error"]]
        exact = sum(1 for x in ok if x["wer"] == 0)
        mean = sum(x["wer"] for x in ok) / len(ok) if ok else float("nan")
        print(f"{tts_name}: {exact}/{len(r)} sentences exact, mean WER {mean:.3f}, "
              f"errors {len(r) - len(ok)}")
        failed |= exact < len(r)
    print(f"Wrote {out}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--tts", default="openai,gemini", help="comma list: openai, gemini")
    ap.add_argument("--stt", default="openai", help="provider used to check: openai or gemini")
    ap.add_argument("--out", default="bench_results/tts_check")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
