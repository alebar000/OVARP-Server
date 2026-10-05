# OVARP-Server

Open Virtual Agent Research Platform — FastAPI server that orchestrates embodied virtual agents
for XR/web HCI experiments. Sits between AI providers (OpenAI / Gemini / any OpenAI-compatible
endpoint) and clients (Unity WebGL, XREAL, Unreal, web), and ships a Wizard-of-Oz console for
researchers.

Companion repos (additional working directories):

- `../OVARP-UnityWebClient` — Unity 6 WebGL reference client.

**This repository (`alebar000/OVARP-Server`) is the definitive one.** It integrates the
SpatialLab-UCENFOTEC line, which no longer receives separate development. Push here.

## Environment

Python 3.10+ is **required** (`str | Path` annotations evaluate at import time).
The machine default `python3` is 3.9.6 and will fail on import. Use `/opt/homebrew/bin/python3.11`.

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt -r requirements_dev.txt
```

`.env` holds the API keys and the optional security settings — see `.env.example`.

## Commands

```bash
# Run (full WoZ console at http://localhost:8000)
./venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run headless (lightweight dashboard, no 3D avatar)
OVARP_HEADLESS=true ./venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000

# Tests — OVARP_TESTING=1 skips transport/provider bootstrap and serves a bare app
OVARP_TESTING=1 OPENAI_API_KEY=sk-dummy GEMINI_API_KEY=dummy ./venv/bin/python -m pytest -q

# Lint
./venv/bin/python -m ruff check src/ tests/
```

Expose the server to a phone or another machine with a tunnel; a browser page served over HTTPS
cannot open a `ws://` socket, so LAN IPs only work for native XR builds:

```bash
cloudflared tunnel --url http://localhost:8000     # then use the wss:// form of the printed URL
```

## Architecture

Message bus, not a request/response API. Everything on the wire is a `BaseCommand`.

```
Transports (ZMQ 5555/5556, WS /ws/client/{id})
    ↓ raw JSON
CommandRouter (src/core/router.py)          — validate → telemetry → dispatch
    ↓ intercepts by (command_type, command)
DialogOrchestrator (src/core/orchestrator.py) — STT → LLM → TTS, per-agent state
    ↓ providers/ (openai | gemini | custom, all OpenAI-compatible)
back through Router → dispatch_outbound → all transports
```

### Layout

- **`src/main.py`** — bootstrap only: build the orchestrator, wire the runtime, mount routers.
  Route handlers live in `src/api/routers/`; adding an endpoint means editing a router, not this.
- **`src/core/runtime.py`** — the composition root. Routers read `runtime.orchestrator`,
  `runtime.telemetry`, etc. rather than importing singletons, which is also how tests substitute
  them (`monkeypatch.setattr(runtime, "orchestrator", mock)`).
- **`src/core/schemas.py`** — `BaseCommand`. Its `target_device`, `target_agent` and `subcommand`
  validators check against `config.yaml` at runtime. **A device, agent, or a value of a declared
  command category that is missing from `config.yaml` is a hard `ValidationError`**, not a warning.
  An `execute_state` may only name declared categories: an unknown *category key* is rejected too,
  so a typo in a category name fails loudly. Other commands keep free-form subcommands.
- **`src/core/config.py`** — `config_manager` singleton; `config.yaml` defines the experiment
  vocabulary. Changing it changes what the schema accepts and what the WoZ UI renders.
- **`src/core/orchestrator.py`** — per-agent state (`_agent_state`): prompt, history, voice.
- **`src/core/profile_manager.py`** — YAML personas in `profiles/`, persisted on create/update.
- **`src/core/survey_manager.py`** — questionnaires in `surveys/`, with SUS and UEQ scoring.
- **`src/core/secrets.py`** — encrypts provider credentials at rest, redacts them from responses.
- **`src/core/key_store.py`** — the provider keys the console can set at runtime. Values live in
  `provider_keys.yaml` (git-ignored) through `secrets.py`, are mirrored into `os.environ`, and
  are restored on boot, where a stored key **wins over `.env`**.
- **`src/api/deps.py`** — the console access token.
- **`src/static/index.html`** — the WoZ console shell, built on Alex's design system v3.0
  (CSS custom properties per theme, `.card` / `.metric-box` / `.badge-*` / `.btn-ghost` /
  `.btn-action-primary`, monospace pill nav). Colours come from those tokens, not from Bootstrap
  semantic classes, and the markup carries **no emoji** — that was a deliberate sweep on both
  sides. New behaviour goes in `src/static/js/*.js` modules, which the shell imports; they talk
  to it through DOM ids and `document` events.
  Tabs are numbered in study order: `(01) OVERVIEW` … `(08) SYSTEM LOGS`. Overview is the
  landing tab and must stay that way — opening on a control surface was the single loudest
  usability complaint.

### Single sources of truth

Several settings used to be reachable two ways, with one silently winning:

- **Prompt** — the Playground writes the global prompt, a profile writes the agent's own, and the
  agent's wins. `/api/llm/config` reports `agents_overriding_global` so the console can say so, and
  `POST /api/agents/{id}/reset` releases an agent back to the global one.
- **Voice** — `_resolve_tts(agent_id)` decides what an agent speaks with. A profile pins it; anything
  unpinned follows the console's picker.
- **Conditions** — legacy `conditions` in `config.yaml` migrate to `condition_*` profiles at boot.
  The scenario runner resolves a step's `condition` through the profile system, so it and the
  Profiles tab take the same code path.

### Wire protocol

Client → server, raw `BaseCommand` JSON:

| command_type | command | subcommand | effect |
|---|---|---|---|
| `audio` | `stt_request` | `{audio_base64}` | STT → LLM → TTS pipeline; not rebroadcast |
| `message` | `llm_request` | `{text}` | LLM → TTS pipeline |
| `message` | `direct_tts` | `{text}` | WoZ text spoken verbatim, bypasses the LLM |
| `system` | `log_marker` | `{label, metadata}` | timestamped event marker |
| `action` | `execute_state` | see below | forwarded to clients |

Server → client, **wrapped by the WS transport**: `{"topic": <command_type>, "payload": <BaseCommand>}`.
ZMQ sends the bare command with a topic frame. Outbound: `user_transcript`, `llm_reply`,
`tts_chunk` (base64 WAV slices), `tts_complete`, `execute_state`, `marker_logged`.

`execute_state` subcommand keys map 1:1 to `config.yaml → custom_commands`:
`emotions`, `actions`, `looks`, `movement`, `avatar`.

### Routing gotchas

- `sender` is **not** validated; `target_device` and `target_agent` **are**.
- For `stt_request` the router replies to `command.sender` as the target device. If that sender ID
  is not a registered device, the pipeline raises inside a fire-and-forget task and dies silently.
- `llm_reply` and `tts_chunk` are always broadcast to `all` so the WoZ console hears them too;
  only `execute_state` is targeted.

### Telemetry is append-only

`data/sessions/*.jsonl` is the record of what happened live. Editing a marker never rewrites the
original entry — it appends `marker_amended` with the previous value. Keep that property when
adding anything that lets a researcher change recorded data.

What goes in: every command the router dispatches (`event=interaction`), and every request a
client sends to the pipeline — `stt_request`, `llm_request`, `direct_tts` — as `event=inbound`,
logged before it is handled. The participant's raw audio is never stored: `log_inbound` replaces
`audio_base64` with `audio_bytes`, and the words arrive later as `user_transcript`. Marker entries
carry `marker_id`, the same id `marker_amended` / `marker_deleted` use, so a correction can be
joined to what it corrects.

## Security

Both are opt-in via `.env`, and absent by default so a localhost run needs no setup:

- `OVARP_ACCESS_TOKEN` — required in the `X-OVARP-Token` header on every `/api/` call. The agent
  WebSocket and the participant survey page are exempt: XR clients and participants have no token.
- `OVARP_SECRET_KEY` — encrypts stored credentials: custom providers in `custom_providers.yaml`
  and the built-in provider keys in `provider_keys.yaml`. Without it both keep working in plain
  text, so an existing localhost setup does not break on upgrade.

Set both before exposing the server through a tunnel.

Credentials never ride along on a response that something polls. `GET /api/providers` reports
`has_key`, `GET /api/keys/status` reports a mask and a badge, and the raw value is only returned
by `POST /api/keys/reveal`, which the console calls when a researcher presses **Show**. Never add
`full_key` to a status response.

## Unity client contract (`../OVARP-UnityWebClient`)

- `Assets/Scripts/OvarpServerConnector.cs` — WebSocket client (NativeWebSocket). Parses the
  `topic`/`command` envelope with a hand-rolled string extractor, not a JSON parser: **field
  names and nesting must stay flat and stable**.
- Identifies itself as `web_01`, baked into the build. That ID must exist in `config.yaml → devices`.
- Audio only — the Unity client never sends `llm_request`. Text injection is WoZ-console-only.
- A page served over HTTPS can only reach `wss://`. **`localhost` no longer works either**:
  current Chrome blocks a public HTTPS origin from opening a socket to the loopback address with
  `net::ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS`. Verified 2026-09-28 against the Vercel build.
  Testing the published client against a local server needs a tunnel, or Chrome started with
  `--disable-features=LocalNetworkAccessChecks,PrivateNetworkAccessChecks`. `/api/server/info`
  returns `public_ws_url` (scheme-aware, honours `X-Forwarded-Proto`) and `lan_ws_url` separately.
- Vocabulary drift to watch: the client's `GesturePlayer` supports `thumbs_up`, absent from
  `config.yaml → custom_commands.actions` and rejected server-side.
- Avatar swap (`OnAgentAvatarChange`) is a logged TODO in `Controller.cs`, not implemented.

## Conventions

- Naming migrated OAF → OVAF → OVARP. `OAF_TESTING` / `OAF_HEADLESS` still work with a deprecation
  warning; use `OVARP_*`. Loggers are namespaced `OVARP.<module>` and do not propagate to root.
- Dual logging on purpose: `structlog` for machine-readable telemetry, stdlib `logging` for the
  human/WS log stream. Follow whichever the surrounding module uses.
- A new provider implements `BaseLLMProvider.generate_response_with_actions` returning
  `(spoken_reply, actions_dict)`. Anything OpenAI-compatible needs no code — register it at
  `/api/providers/register`.
- **Provider SDK clients are lazy `@property` reads, never bound in `__init__`.** The console can
  change an API key at runtime; an eagerly bound client would keep authenticating with the old
  one. `OpenAIClientSingleton.reset_client()` / `GeminiClientSingleton.reset_client()` drop the
  cache, and `key_store.set_keys()` calls both. This regressed once already — see below.
- Model ids come from env vars with defaults: `OVARP_GEMINI_{LLM,STT,TTS}_MODEL` and
  `OVARP_OPENAI_{LLM,STT,TTS}_MODEL` (see `.env.example`).
- Sentence-level path: gestures/emotion from `extract_actions` are dispatched the moment the
  classification returns, concurrently with speech. A failed `stream_reply` raises; the
  orchestrator drains the sentences already queued and reports `pipeline_error`.
- `/api/health/providers` checks **key presence, not connectivity**. Probing the providers for
  real took 10s+ and left the console showing stale error badges while it waited.
- Functions used from inline `onclick=` must be assigned to `window` explicitly: the console runs
  as a `<script type="module">`, where declarations are not global.
- **Route order decides whether a route is protected.** `surveys.results_router` is mounted before
  `surveys.router` in `main.py` because that router ends in a catch-all `/api/surveys/{survey_id}`.
  Mounted the other way round, `/api/surveys/responses` matches the catch-all and hands participant
  data out with no token. `test_security.py` pins this; it shipped wrong once.
- **An `execute_state` from the LLM carries every category at once.** The tool schema marks each
  one `required`, so every declared category arrives together on each turn. A client that tests
  them with `else if` only ever reacts to the first — which is why the Unity avatar looked
  unimplemented for three QA rounds when it was not.
- **A category with one value is not a choice, and both layers treat it that way.** The LLM tool
  schema offers it but does not `require` it (every schema the server builds: both providers'
  `_build_tools_schema`, Gemini's `_build_actions_only_tools_schema` and the OpenAI-compatible
  custom provider), and WoZ
  Control does not render a button row for it at all — a control to "change" something with one
  option promises what cannot happen. `avatar` is in that state: declared with one value
  (`default`) because the Unity client cannot swap the model yet. Add a second value and both the
  requirement and the button row come back on their own. Neither is a carve-out for avatar.
- Nothing dispatches an avatar `execute_state`. `AgentProfile.avatar` is recorded but inert, and
  the Unity client has no `OnAvatarCommand`; restore both alongside the extra config values when
  the prefab swap lands.
- Ids that become filenames go through `src/core/identifiers.py`. `profile_manager` and
  `scenario_runner` both write `<dir>/<id>.yaml` from a value that arrives over the API.

## Known state (v1.0.0)

Tests: 337 passing. `ruff check` is clean on `src/api/`, `src/main.py`, `src/core/key_store.py`
and the modules added recently; the older files still carry ~400 violations (whitespace, line
length, `Optional[X]`), not gated in CI. New code in an older file matches that file's existing
style rather than importing a second convention into it.

## Settled design decisions

Do not re-litigate these:

- **The routers architecture stays.** `src/api/routers/` + `runtime.py`; no monolithic
  `main.py`.
- **The console design** — design system, tab order, Overview, API Key Store, Surveys tab and
  `/player` — is the one in `src/static/index.html`.
- **API keys go through the encrypted store** (`secrets.py`, `key_store.py`), never plaintext
  `.env` writes, and the key status response never returns `full_key` (see Security).
- **Markers** are amended with `PATCH /api/session/marker/{id}` and retracted with `DELETE`;
  both are logged as new telemetry events, never rewritten in place. There is no `PUT`.
- **One client SDK**: `OVARP-client.js`. Do not add a second one.
- **Avatars must be VRM** — `avatar.js` loads through `VRMLoaderPlugin`, so FBX/GLB demo
  characters (e.g. the old `friet256`) are not shipped.
- `web_01` and `xreal_01` must stay declared in `config.yaml`, or the Unity client is rejected.

Pre-existing, not things to fix unprompted:

- `venv/` is broken — its interpreter points at the pre-rename path
  `/Users/briammora/Projects/github/OpenVirtualAgentFramework-Server/venv`. Recreate it, or
  go through `./venv/bin/python -m pip` / `-m pytest`, which ignores the shebang.
- `src/woz/` and `src/base/` are empty packages left from an earlier layout.
