# Original User Request

## Initial Request - 2026-08-07T21:55:37Z

OVARP-Server Web UI Redesign, Multi-Platform Live Control, Event Reclassification, Internal Usability Evaluation (SUS/UEQ), and Secure Key Storage.

Working directory: c:/AntiGravityStuff/OVARP
Integrity mode: development

## Requirements

### R1. UI Aesthetics, Light/Dark Theme & Active State Indicators
Implement a modern, polished web interface with global Light / Dark mode toggle (persisted in localStorage). Ensure high-contrast visual indicators across the UI for: active tab, active LLM provider badge, active TTS voice, active Profile, and API key storage state. Write code with NO em-dashes.

### R2. Overview Dashboard & Standalone Web Player (OPEN PLAYER)
Set the Overview Dashboard as the default landing tab with quick-start cards and server telemetry status. Create src/static/player.html as an independent 3D VRM agent player with microphone STT, TTS audio lip-sync, and chat. Add a prominent "OPEN PLAYER" button in the WoZ Remote Live Agent Control tab (renamed from Remote Agent Control). Write code with NO em-dashes.

### R3. LLM Playground & Profile Management Integration
Integrate profile selection and inline "+ New Profile" creation directly into the LLM Playground. Display a high-visibility status card showing the active LLM provider, active model ID, TTS engine, and STT/LLM/TTS latency breakdown. Enable full profile creation, duplication, and editing in the Profiles tab. Write code with NO em-dashes.

### R4. Study Session Event Marker Reclassification
Enhance SessionManager and main.py with PUT /api/session/markers/{marker_id} to allow editing logged event markers (notes, labels, and category reclassification such as changing "Participant Confused" to "Technical Issue"). Update the UI with edit/reclassify controls. Write code with NO em-dashes.

### R5. Internal OVARP Usability Evaluation Module (SUS / UEQ)
Implement a dedicated internal evaluation module/tab for researchers to assess OVARP. Include a 10-item System Usability Scale (SUS), a User Experience Questionnaire (UEQ) scale, and qualitative feedback questions (usage intent, target use-cases, improvements, pros/cons). Store results via POST /api/evaluations/ovarp into data/evaluations/. Write code with NO em-dashes.

### R6. Secure API Key Store with Visual Storage State
Implement an API Key Store tab allowing researchers to enter API keys for OpenAI, Gemini, and ElevenLabs. Keys must update in memory immediately with an explicit "Persist to .env" option. Display clear, high-visibility status badges ([GUARDADO EN .ENV] vs [SOLO EN MEMORIA - NO GUARDADO]) and validation status. Write code with NO em-dashes.

### R7. No Git Check-ins / Commits Prior to Review
Do NOT make any git commit, git push, or version control check-ins during execution. Keep all modified and new files uncommitted in the working tree until explicit user review.

---

## Acceptance Criteria

### Automated Tests
- pytest runs cleanly with 100% pass rate across all existing tests and new tests for API keys, marker editing, and evaluation endpoints.
- PUT /api/session/markers/{marker_id} updates marker category and notes in session_<id>.jsonl.
- POST /api/evaluations/ovarp saves valid evaluation payloads and computes SUS scores correctly.

### Interactive Agent-as-User Verification (Mandatory)
- An independent tester agent launches the OVARP server and interacts directly with the live endpoints and UI components end-to-end to verify functionality as a real user would.
- Tester agent verifies theme switching (Light/Dark mode) and active tab indicators.
- Tester agent verifies creating a new profile from the LLM Playground, sending a message, and observing the active provider status badge.
- Tester agent verifies starting a session, adding a marker, reclassifying the marker category, and filling out the SUS/UEQ evaluation form.
- Tester agent verifies opening player.html (OPEN PLAYER) and checking interactive avatar/chat capabilities.
- Tester agent verifies API key storage status badges ([GUARDADO EN .ENV] vs [SOLO EN MEMORIA]).

### Git Working Tree Verification
- All code changes remain in working directory without any git commits performed.
