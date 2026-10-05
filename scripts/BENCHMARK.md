# OVARP latency benchmark: how to run it

This protocol produces the numbers for the performance section of the JSS paper:
time to first audio (TTFA) under full-reply synthesis versus sentence-level TTS,
per provider pair and reply length. It runs the real `DialogOrchestrator`
in-process, so it needs no browser or headset. Times are taken where each
message leaves the server.

## 1. What you need

- The OVARP-Server repo at the commit you want to report. Write down the hash, or let
  the script record it in `env.json`.
- Python 3.10+ with the server dependencies, plus `scipy` for the significance test:
  `pip install -r requirements.txt scipy`
- Keys in `.env`: `GEMINI_API_KEY` (required) and `OPENAI_API_KEY` (for the OpenAI
  pairs). Leave the model variables unset unless you mean to change them. The defaults
  get recorded.
- A stable wired connection. Run all conditions in one session from the same machine.
  Don't run it over a hotspot.

## 2. Check the harness first (about 2 min, no keys)

```bash
python scripts/bench_latency.py --mock --reps 2 --out bench_results
```

You should get a summary table. In the `long` class, `sentence` should come out much
lower than `full`. This step tests only the script, not the providers.

## 3. The real run (about 40-60 min)

```bash
python scripts/bench_latency.py \
  --pairs gemini:gemini,gemini:openai,openai:openai \
  --reps 10 --gap 2.0 --warmup 2 \
  --notes "machine=<model>, network=<wired/campus>, location=<city>, date=<yyyy-mm-dd>" \
  --out bench_results
```

- `gemini:*` runs both modes (`full`, `sentence`). `openai:openai` runs only `full`,
  because the OpenAI LLM provider has no `stream_reply`. That's expected, and the
  paper says so.
- 12 prompts × 10 reps × 2 modes = 240 turns per Gemini pair; 120 for the OpenAI pair.
- The run shuffles prompt order and alternates which mode goes first (ABBA), so slow
  drift in provider latency doesn't favour either mode.
- If you hit rate limits, raise `--gap` to 4. If time is short, run `--reps 5` now and
  more later. Runs can be merged.

## 4. What to send back

Zip the whole `bench_results/<run_id>/` folder:

| file | contents |
|---|---|
| `trials.csv` | one row per turn: all timings, reply length, audio seconds, modeled stalls, errors |
| `summary.md` / `summary.csv` | medians, IQR and p90 per condition and length class |
| `paired.csv` | paired effect of sentence vs. full on the same prompt and rep: median difference, bootstrap 95% CI, Wilcoxon p |
| `env.json` | commit hash, models, voices, Python, OS, arguments, notes |

## 5. Optional

- `--mock-zero --reps 25` measures OVARP's own overhead with zero-latency providers.
  The paper already reports one such run. Repeat it on the lab machine if you want
  the number from that hardware.
- `--summarize bench_results/<run_id>` re-runs the analysis without new API calls.

## Metric definitions

- **TTFA**: time from submitting the user's text to the first `tts_chunk` leaving the server.
- **done**: time to the last `tts_complete` (all audio sent).
- **speech_end** (modeled): when a client playing utterances in order finishes speaking.
  Uses the WAV durations.
- **stall**: modeled silence between two sentences while the next sentence's audio
  hasn't arrived yet. Happens only in sentence mode.
- **srv_\***: the server's own latency record for the turn (`stt/llm/tts/total`),
  the same values the console and the session JSONL show.

STT is out of scope here: turns are submitted as text, so speech recognition doesn't
add variance. To include STT, run the `latency_validation` scenario from the console.
That writes `event=latency` rows with `stt_ms` to the session JSONL.
