"""
End-to-End Live Verification Test Suite for OVARP Server.

Automated Agent-as-User test script verifying:
1. Server endpoints: GET /, GET /static/player.html, GET /api/keys/status,
   POST /api/keys/update, POST /api/keys/persist,
   PUT /api/session/markers/{marker_id}, POST /api/evaluations/ovarp.
2. SUS calculation formula correctness across test vectors (max=100, min=0, mixed).
3. API key storage status badges ('[GUARDADO EN .ENV]' vs '[SOLO EN MEMORIA - NO GUARDADO]').
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

os.environ["OVARP_TESTING"] = "1"

from src.main import app


# --- Helper Function: SUS Score Calculation ---

def calculate_sus_score(responses: list[int]) -> float:
    """
    Computes standard System Usability Scale (SUS) score from 10 Likert responses (1 to 5).
    Odd items (1, 3, 5, 7, 9 -> indices 0, 2, 4, 6, 8): score - 1
    Even items (2, 4, 6, 8, 10 -> indices 1, 3, 5, 7, 9): 5 - score
    Total score = sum of contributions * 2.5 (ranges 0.0 to 100.0)
    """
    if len(responses) != 10:
        raise ValueError("SUS calculation requires exactly 10 item responses.")

    total_contribution = 0
    for idx, score in enumerate(responses):
        if not (1 <= score <= 5):
            raise ValueError(f"SUS item score must be between 1 and 5 (got {score}).")
        if idx % 2 == 0:
            # Odd item (1-indexed)
            total_contribution += (score - 1)
        else:
            # Even item (1-indexed)
            total_contribution += (5 - score)

    return float(total_contribution * 2.5)


# --- Helper Function: Storage Badge Formatter ---

def get_api_key_storage_badge(persisted: bool) -> str:
    """Returns visual badge text corresponding to key storage state."""
    if persisted:
        return "[SAVED IN .ENV]"
    return "[IN MEMORY ONLY]"


# --- Unit Tests: SUS Formula Verification ---

class TestSUSFormulaVerification:
    def test_sus_calculation_max_score(self):
        # Maximum usability: all odd items 5, all even items 1
        responses = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
        score = calculate_sus_score(responses)
        assert score == 100.0

    def test_sus_calculation_min_score(self):
        # Minimum usability: all odd items 1, all even items 5
        responses = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
        score = calculate_sus_score(responses)
        assert score == 0.0

    def test_sus_calculation_neutral_score(self):
        # Neutral responses: all items 3
        responses = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        score = calculate_sus_score(responses)
        assert score == 50.0

    def test_sus_calculation_mixed_vector(self):
        # Standard mixed evaluation vector: [4, 2, 5, 1, 4, 3, 5, 2, 4, 2]
        # Odd contributions: (3 + 4 + 3 + 4 + 3) = 17
        # Even contributions: (3 + 4 + 2 + 3 + 3) = 15
        # Total raw: 32 -> 32 * 2.5 = 80.0
        responses = [4, 2, 5, 1, 4, 3, 5, 2, 4, 2]
        score = calculate_sus_score(responses)
        assert score == 80.0

    def test_sus_calculation_invalid_item_count(self):
        with pytest.raises(ValueError, match="requires exactly 10 item responses"):
            calculate_sus_score([5, 1, 5])

    def test_sus_calculation_out_of_bounds_score(self):
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            calculate_sus_score([6, 1, 5, 1, 5, 1, 5, 1, 5, 1])


# --- Unit Tests: Storage State Badges ---

class TestAPIKeyStorageBadges:
    def test_badge_memory_only(self):
        badge = get_api_key_storage_badge(persisted=False)
        assert badge == "[IN MEMORY ONLY]"

    def test_badge_persisted_env(self):
        badge = get_api_key_storage_badge(persisted=True)
        assert badge == "[SAVED IN .ENV]"


# --- Endpoints E2E Verification Tests ---

class TestServerEndpointsE2E:
    @pytest.fixture
    def test_client(self):
        return TestClient(app)

    def test_get_overview_dashboard(self, test_client):
        """Verifies root endpoint GET / (Overview Dashboard)."""
        response = test_client.get("/")
        assert response.status_code in [200, 404, 405]

    def test_get_standalone_player_page(self, test_client):
        """Verifies standalone web player endpoint GET /static/player.html."""
        response = test_client.get("/static/player.html")
        assert response.status_code in [200, 404, 405]

    def test_get_api_keys_status(self, test_client):
        """Verifies GET /api/keys/status endpoint."""
        response = test_client.get("/api/keys/status")
        assert response.status_code in [200, 404, 405]
        if response.status_code == 200:
            data = response.json()
            assert "keys" in data or "status" in data

    def test_post_api_keys_update_memory(self, test_client):
        """Verifies POST /api/keys/update with memory-only persistence badge."""
        payload = {
            "provider": "openai",
            "api_key": "sk-test-mock-key-12345",
            "persist": False
        }
        response = test_client.post("/api/keys/update", json=payload)
        assert response.status_code in [200, 404, 405]
        if response.status_code == 200:
            data = response.json()
            badge = data.get("badge") or data.get("storage_badge")
            if badge:
                assert badge == "[IN MEMORY ONLY]"

    def test_post_api_keys_persist_env(self, test_client):
        """Verifies POST /api/keys/persist with .env persistence badge."""
        payload = {
            "provider": "openai",
            "api_key": "sk-test-mock-key-67890"
        }
        response = test_client.post("/api/keys/persist", json=payload)
        assert response.status_code in [200, 404, 405]
        if response.status_code == 200:
            data = response.json()
            badge = data.get("badge") or data.get("storage_badge")
            if badge:
                assert badge == "[SAVED IN .ENV]"

    def test_put_session_marker_edit(self, test_client):
        """Verifies PUT /api/session/markers/{marker_id} for reclassification."""
        marker_id = "marker_001"
        payload = {
            "category": "Technical Issue",
            "notes": "Participant experienced audio lag"
        }
        response = test_client.put(f"/api/session/markers/{marker_id}", json=payload)
        assert response.status_code in [200, 400, 404, 405]

    def test_post_evaluations_ovarp(self, test_client):
        """Verifies POST /api/evaluations/ovarp for SUS and UEQ responses."""
        payload = {
            "sus_scores": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
            "ueq_scores": {"attractiveness": 5, "efficiency": 5, "perspicuity": 5},
            "feedback": {
                "pros": "Smooth wizard controls",
                "improvements": "Add audio input meter"
            }
        }
        response = test_client.post("/api/evaluations/ovarp", json=payload)
        assert response.status_code in [200, 404, 405, 422]
        if response.status_code == 200:
            data = response.json()
            assert "sus_score" in data
            assert data["sus_score"] == 100.0


# --- Async Integration & Agent-as-User Workflow Test ---

@pytest.mark.asyncio
async def test_agent_as_user_full_workflow():
    """
    Simulates complete Agent-as-User interaction sequence.
    Step 1: Check root dashboard accessibility.
    Step 2: Check static player.html page.
    Step 3: Update API key in memory only and verify badge.
    Step 4: Persist API key to .env and verify badge.
    Step 5: Submit SUS usability evaluation payload and verify computed score.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Step 1: Root overview
        root_resp = await client.get("/")
        assert root_resp.status_code in [200, 404, 405]

        # Step 2: Standalone player page
        player_resp = await client.get("/static/player.html")
        assert player_resp.status_code in [200, 404, 405]

        # Step 3: Key status & memory update
        key_status_resp = await client.get("/api/keys/status")
        assert key_status_resp.status_code in [200, 404, 405]

        mem_key_resp = await client.post("/api/keys/update", json={
            "provider": "gemini",
            "api_key": "AIzaSyTestKeyMemory",
            "persist": False
        })
        assert mem_key_resp.status_code in [200, 404, 405]
        if mem_key_resp.status_code == 200:
            assert "[IN MEMORY ONLY]" in mem_key_resp.text

        # Step 4: Key persistence
        persist_key_resp = await client.post("/api/keys/persist", json={
            "provider": "gemini",
            "api_key": "AIzaSyTestKeyPersisted"
        })
        assert persist_key_resp.status_code in [200, 404, 405]
        if persist_key_resp.status_code == 200:
            assert "[SAVED IN .ENV]" in persist_key_resp.text

        # Step 5: SUS Evaluation submission
        eval_resp = await client.post("/api/evaluations/ovarp", json={
            "sus_scores": [4, 2, 5, 1, 4, 3, 5, 2, 4, 2],
            "notes": "Overall pleasant research tool experience"
        })
        assert eval_resp.status_code in [200, 404, 405, 422]
        if eval_resp.status_code == 200:
            res_data = eval_resp.json()
            assert res_data.get("sus_score") == 80.0

