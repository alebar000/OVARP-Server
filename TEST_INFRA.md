# OVARP Test Infrastructure Documentation

## 1. Overview and Purpose

This document specifies the automated testing architecture, feature coverage checklist, execution procedures, and pass criteria for the Open Virtual Agent Research Platform (OVARP). The test suite ensures reliable core business logic, robust REST API endpoints, real-time telemetry streaming, secure key management, and end-to-end user workflows.

## 2. Four-Tier Test Architecture

The OVARP test framework is organized into four complementary testing tiers:

```
+-------------------------------------------------------------------+
|               TIER 4: Performance & Telemetry Validation          |
|  - XR Telemetry streaming high-frequency load (ZMQ/WS)            |
|  - Memory stability and latency breakdown tracking                |
+-------------------------------------------------------------------+
                                  ^
                                  |
+-------------------------------------------------------------------+
|            TIER 3: Interactive Agent-as-User E2E Harness          |
|  - Live server process execution & UI workflow simulation         |
|  - End-to-end user journeys (Theme, Profiles, Session, SUS, Keys) |
+-------------------------------------------------------------------+
                                  ^
                                  |
+-------------------------------------------------------------------+
|               TIER 2: FastAPI REST API & Integration Tests        |
|  - FastAPI TestClient / httpx AsyncClient route validation        |
|  - Marker editing, SUS evaluation post, Key store memory/.env     |
+-------------------------------------------------------------------+
                                  ^
                                  |
+-------------------------------------------------------------------+
|                  TIER 1: Core Isolated Unit Tests                 |
|  - Pydantic schema validation, SUS mathematical formula           |
|  - SessionManager, ProfileManager, ConfigManager unit tests       |
+-------------------------------------------------------------------+
```

### Tier 1: Core Isolated Unit Tests
- Location: `tests/core/`, `tests/test_schemas.py`, `tests/test_session_manager.py`, `tests/test_profile_manager.py`
- Focus: Mathematical logic (SUS calculation formula), Pydantic schema parsing/validation, profile YAML parsing, state transitions.
- Execution Time: < 0.5s
- Dependencies: None (pure Python and memory objects).

### Tier 2: REST API & Integration Tests
- Location: `tests/test_api_*.py`, `tests/test_main.py`
- Focus: Route handling using FastAPI TestClient / httpx AsyncClient. Validates HTTP status codes, JSON payload responses, disk file persistence in `data/evaluations/` and `data/sessions/`.
- Execution Time: < 2.0s
- Dependencies: FastAPI application context.

### Tier 3: Interactive Agent-as-User E2E Harness
- Location: `tests/test_e2e_live_verification.py`
- Focus: Simulates end-to-end user interactions against live running server endpoints and static assets. Tests Overview dashboard, Standalone Web Player, marker reclassification (`PUT /api/session/markers/{marker_id}`), SUS/UEQ evaluation submission (`POST /api/evaluations/ovarp`), and API key storage state badges (`[GUARDADO EN .ENV]` vs `[SOLO EN MEMORIA - NO GUARDADO]`).
- Execution Time: ~3-5s
- Dependencies: Subprocess or TestClient environment with full route set.

### Tier 4: Performance & Telemetry Validation
- Location: `tests/test_xr_telemetry.py`
- Focus: High-frequency telemetry packet processing over ZeroMQ and WebSockets, message buffer stability, latency telemetry logging.
- Execution Time: < 1.0s
- Dependencies: Transport layers.

## 3. Feature Coverage Checklist

| Feature Requirement | Primary Endpoint / File | Target Coverage | Key Test Verification Points |
|---|---|---|---|
| R1: Light/Dark Theme & Active States | UI CSS / LocalStorage / GET / | 100% | Contrast badges, localStorage persistence keys, active provider badges |
| R2: Overview & Standalone Web Player | `GET /`, `GET /static/player.html` | 100% | Player page HTML resolution, WebRTC/STT/TTS control hooks, navigation links |
| R3: LLM Playground & Profile Management | `/api/profiles`, `/api/llm/config` | 100% | Inline profile creation, active provider badge status, latency breakdown |
| R4: Marker Editing & Reclassification | `PUT /api/session/markers/{marker_id}` | 100% | Marker category/note updates, real-time disk persistence to `session_<id>.jsonl`, 404/400 error handling |
| R5: Internal Usability Module (SUS/UEQ) | `POST /api/evaluations/ovarp` | 100% | Standard 10-item SUS formula computation (Max 100.0, Min 0.0), UEQ scores, JSON persistence to `data/evaluations/` |
| R6: Secure API Key Store | `/api/keys/status`, `/api/keys/update`, `/api/keys/persist` | 100% | In-memory key application (`[SOLO EN MEMORIA - NO GUARDADO]`), `.env` persistence (`[GUARDADO EN .ENV]`), key masking |

## 4. SUS Score Calculation Formula Specification

The System Usability Scale (SUS) calculation follows the standard Brooke (1996) methodology:
- Input: 10 Likert-scale items scored 1 to 5.
- Odd items (1, 3, 5, 7, 9): Score contribution = `score - 1`.
- Even items (2, 4, 6, 8, 10): Score contribution = `5 - score`.
- Sum of all 10 item contributions multiplied by 2.5 yields the final SUS score (0.0 to 100.0).

### Validation Test Vectors
1. Perfect Score: All 5s -> `(5-1)*5 + (5-5)*5` = 20 + 20 = 40 raw points * 2.5 = 100.0
2. Lowest Score: All 1s -> `(1-1)*5 + (5-1)*5` = 0 + 0 = 0 raw points * 2.5 = 0.0
3. Neutral Score: All 3s -> `(3-1)*5 + (5-3)*5` = 10 + 10 = 20 raw points * 2.5 = 50.0
4. Mixed Vector `[4, 2, 5, 1, 4, 3, 5, 2, 4, 2]`:
   - Odd contributions: (4-1) + (5-1) + (4-1) + (5-1) + (4-1) = 3 + 4 + 3 + 4 + 3 = 17
   - Even contributions: (5-2) + (5-1) + (5-3) + (5-2) + (5-2) = 3 + 4 + 2 + 3 + 3 = 15
   - Total raw score: 17 + 15 = 32 points * 2.5 = 80.0

## 5. API Key Storage Badges Specification

- Memory Only Mode (`persist=False`):
  - In-memory configuration is updated immediately for runtime calls.
  - `.env` file is unchanged.
  - Status badge string: `[SOLO EN MEMORIA - NO GUARDADO]`
- Persisted Mode (`persist=True` or via `/api/keys/persist`):
  - In-memory configuration is updated immediately.
  - Key is saved to `.env` file.
  - Status badge string: `[GUARDADO EN .ENV]`

## 6. Execution Commands and Pass Criteria

To execute the complete test suite:

```bash
pytest -v
```

To run with coverage reporting:

```bash
pytest --cov=src --cov-report=term-missing
```

### Pass Criteria
- 100% test pass rate across all collected items.
- No unhandled exceptions or 500 status codes during endpoint verification.
- Validated mathematical accuracy for SUS calculations.
- Confirmed file writes in `data/evaluations/` and `data/sessions/`.
- Absolutely no em-dashes in source or test code strings.
