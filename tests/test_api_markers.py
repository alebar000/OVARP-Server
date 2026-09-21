"""
Unit and Integration Tests for Event Markers (R4 Backend).

Tests the EventMarker model, SessionManager marker reclassification and notes,
and the PATCH /api/session/marker/{marker_id} endpoint.

Reclassification used to be ``update_marker`` behind ``PUT
/api/session/markers/{id}``, which edited the marker in place. It is now
``amend_marker`` behind ``PATCH /api/session/marker/{id}``, which additionally
returns the replaced values so the caller can append them to the session log:
``data/sessions/*.jsonl`` is the record of what happened live, so a correction
is added to it rather than rewriting what was captured. The category, notes and
label a researcher can change are the same.

Author: Alexander Barquero Elizondo, Ph.D. - UCR, ECCI/CITIC
License: MIT
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["OVARP_TESTING"] = "1"

from src.core.session_manager import EventMarker, session_manager
from src.main import app


class TestEventMarkerModel:
    def test_marker_creation_defaults(self):
        marker = EventMarker(timestamp=1000.0, iso_time="2026-08-07T12:00:00", label="task_start")
        assert marker.id is not None
        assert len(marker.id) == 8
        assert marker.category is None
        assert marker.notes is None
        assert marker.amended is False

    def test_marker_creation_full(self):
        marker = EventMarker(
            id="m1234567",
            timestamp=1000.0,
            iso_time="2026-08-07T12:00:00",
            label="confused",
            category="Participant Issue",
            notes="Participant was unsure how to proceed",
            metadata={"step": 2}
        )
        assert marker.id == "m1234567"
        assert marker.category == "Participant Issue"
        assert marker.notes == "Participant was unsure how to proceed"
        assert marker.metadata == {"step": 2}


class TestSessionManagerMarkers:
    def test_add_and_amend_marker_by_id(self):
        if session_manager.is_active:
            session_manager.end_session()

        session_manager.start_session("P_MARKER_01")
        m = session_manager.add_marker(label="initial_label", category="General", notes="Init note")
        assert m.category == "General"

        updated, before = session_manager.amend_marker(
            marker_id=m.id,
            category="Technical Issue",
            notes="Updated note: Lag detected",
            label="reclassified_label"
        )
        assert updated.id == m.id
        assert updated.category == "Technical Issue"
        assert updated.notes == "Updated note: Lag detected"
        assert updated.label == "reclassified_label"

        session_manager.end_session()

    def test_amendment_reports_the_replaced_values(self):
        """What the timestamped log needs in order to stay append-only."""
        if session_manager.is_active:
            session_manager.end_session()

        session_manager.start_session("P_MARKER_02")
        m = session_manager.add_marker(label="first_marker", category="General")

        _, before = session_manager.amend_marker(m.id, category="User Error")

        assert before == {"category": "General"}
        session_manager.end_session()

    def test_amended_marker_is_flagged(self):
        if session_manager.is_active:
            session_manager.end_session()

        session_manager.start_session("P_MARKER_03")
        m = session_manager.add_marker(label="typo")
        assert m.amended is False

        updated, _ = session_manager.amend_marker(m.id, label="fixed")

        assert updated.amended is True
        session_manager.end_session()

    def test_amend_marker_error_no_session(self):
        if session_manager.is_active:
            session_manager.end_session()

        with pytest.raises(ValueError, match="No marker"):
            session_manager.amend_marker("m_invalid", category="Issue")

    def test_amend_marker_error_not_found(self):
        if session_manager.is_active:
            session_manager.end_session()

        session_manager.start_session("P_MARKER_04")
        with pytest.raises(ValueError, match="No marker"):
            session_manager.amend_marker("nonexistent_id", category="Issue")
        session_manager.end_session()


class TestMarkerAPIEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_patch_marker_endpoint_success(self, client):
        if session_manager.is_active:
            session_manager.end_session()

        client.post("/api/session/start", json={"participant_id": "P_API_01"})
        add_resp = client.post("/api/session/marker", json={
            "label": "stuck",
            "category": "Confusion",
            "notes": "User froze"
        })
        assert add_resp.status_code == 200
        marker_id = add_resp.json()["marker"]["id"]

        patch_resp = client.patch(f"/api/session/marker/{marker_id}", json={
            "category": "Technical Issue",
            "notes": "Reclassified: Controller disconnected",
            "label": "hardware_failure"
        })
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert data["status"] == "ok"
        assert data["amended"] is True
        assert data["marker"]["category"] == "Technical Issue"
        assert data["marker"]["notes"] == "Reclassified: Controller disconnected"
        assert data["marker"]["label"] == "hardware_failure"

        client.post("/api/session/end")

    def test_patch_marker_endpoint_no_session(self, client):
        if session_manager.is_active:
            session_manager.end_session()

        patch_resp = client.patch("/api/session/marker/marker_99", json={
            "category": "Issue"
        })
        assert patch_resp.status_code == 200
        assert "error" in patch_resp.json()

    def test_marker_can_be_retracted(self, client):
        if session_manager.is_active:
            session_manager.end_session()

        client.post("/api/session/start", json={"participant_id": "P_API_02"})
        marker_id = client.post(
            "/api/session/marker", json={"label": "fired_by_mistake"}
        ).json()["marker"]["id"]

        delete_resp = client.delete(f"/api/session/marker/{marker_id}")

        assert delete_resp.status_code == 200
        assert delete_resp.json()["status"] == "ok"
        assert client.get("/api/session/status").json()["marker_count"] == 0
        client.post("/api/session/end")
