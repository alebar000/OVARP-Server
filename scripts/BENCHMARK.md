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

## 3. The real run (about 30-40 min)

```bash
python scripts/bench_latency.py \
  --pairs gemini:openai,openai:openai \
  --reps 5 --gap 3 --warmup 2 \
  --notes "machine=<model>, network=<wired/campus>, location=<city>, date=<yyyy-mm-dd>" \
  --out bench_results
```

- `LLM:TTS`. Both pairs use OpenAI's voice (`gpt-realtime-2.1-mini`), so the comparison
  covers both LLM vendors with one voice. It read 12/12 test sentences verbatim and took about
  1 s per sentence, against 2-3.5 s for `gemini-3.8-flash-lite-tts`, which also changed a
  voseo form once (`check_tts_verbatim.py`, 2026-10-13).
- Every pair runs both modes (`full`, `sentence`): all built-in LLM adapters stream.
  The `full` mode hides that capability, so it measures the path an adapter without
  streaming would take.
- 12 prompts × 5 reps × 2 modes = 120 turns per pair, 60 paired comparisons. The effect
  being measured is seconds on long replies, so this is enough; with the default
  (cheapest) models the whole run costs on the order of a dollar or two.
- Run a short check first (`--classes short --reps 1 --warmup 1`): it confirms the keys,
  the models and the rate limits for a few cents.
- To check that a voice reads its text verbatim (it matters for OpenAI's Realtime voice), run
  `python scripts/check_tts_verbatim.py --tts openai,gemini --stt openai`.
- The run shuffles prompt order and alternates which mode goes first (ABBA), so slow
  drift in provider latency doesn't favour either mode.
- If you hit rate limits, raise `--gap` to 5. A turn that fails is recorded as an error
  and left out of the analysis; runs can be merged.

## 4. What to send back

Zip the whole `bench_results/<run_id>/` folder:

| file | contents |
|---|---|
| `trials.csv` | one row per turn: all timings, reply length, audio seconds, modeled stalls, errors |
| `summary.md` / `summary.csv` | medians, IQR and p90 per condition and length class |
| `paired.csv` | paired effect of sentence vs. full on the same prompt and rep: median difference, bootstrap 95% CI, Wilcoxon p |
| `env.json` | commit hash, models, voices, Python, OS, arguments, notes |

## 5. Optional

- `--mock-zero --reps 25 --gap 0` measures OVARP's own overhead: providers are
  replaced by stubs that answer instantly, so every millisecond left is the
  platform's (routing, validation, base64, serialization, the session record and
  the server log). The stubs hand control back to the event loop between streamed
  text chunks, as a network stream does. Takes about 30 s. Report the machine in
  `--notes`; `env.json` records the CPU and whether the fast base64 encoder
  (`pybase64`) was installed.
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
