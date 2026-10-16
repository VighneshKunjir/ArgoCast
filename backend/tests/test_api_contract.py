"""API contract tests for the FastAPI backend."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the startup hook (model + evidence index)
        yield c


@pytest.fixture(scope="module")
def sample_float(client):
    floats = client.get("/api/floats").json()
    return max(floats, key=lambda f: len(f["cycles"]))


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "ArgoCast" in data["service"]


def test_floats_catalog_endpoint(client, sample_float):
    floats = client.get("/api/floats").json()
    assert len(floats) >= 1
    assert len(sample_float["cycles"]) >= 4
    assert sample_float["defaultCycle"] in sample_float["cycles"]


def test_profiles_endpoint(client, sample_float):
    wmo = sample_float["wmo_id"]
    res = client.get(f"/api/profiles/{wmo}")
    assert res.status_code == 200
    profiles = res.json()
    assert len(profiles) == len(sample_float["cycles"])
    assert profiles[0]["wmo_id"] == wmo
    assert len(profiles[0]["measurements"]) == 16


def test_forecast_contract_schema(client, sample_float):
    payload = {"wmoId": sample_float["wmo_id"], "cycle": sample_float["defaultCycle"]}
    res = client.post("/api/forecast", json=payload)
    assert res.status_code == 200
    data = res.json()

    required_keys = [
        "forecast_id", "target_float_id", "target_cycle", "predicted_cycle",
        "profiles", "uncertainty_bounds", "physical_diagnostics",
        "xai_attribution", "evidence_citations", "metrics_comparison",
    ]
    for key in required_keys:
        assert key in data, f"Missing required key '{key}' in forecast response"

    assert len(data["profiles"]) == 16, "Profiles must have 16 standard pressure levels"
    assert len(data["uncertainty_bounds"]) == 16
    assert data["has_observation"] is True  # default base cycle is the second-to-last one
    assert [m["model"] for m in data["metrics_comparison"]] == [
        "Persistence (t-1)", "Ridge regression", "ArgoCast PI-BiLSTM"]
    assert all(p["ridge_temperature"] != 0 for p in data["profiles"])
    for b in data["uncertainty_bounds"]:
        assert b["ci95_temp_upper"] > b["mean_temp"] > b["ci95_temp_lower"]
    assert "is_gravitationally_stable" in data["physical_diagnostics"]
    assert isinstance(data["physical_diagnostics"]["is_gravitationally_stable"], bool)
    if not data["physical_diagnostics"]["is_gravitationally_stable"]:
        assert data["physical_diagnostics"]["stability_violation_count"] > 0
    for cit in data["evidence_citations"]:
        assert cit["qc_flag"] in (1, 2)
        assert cit["wmo_id"] != sample_float["wmo_id"]


def test_forecast_latest_cycle_is_true_lookahead(client, sample_float):
    payload = {"wmoId": sample_float["wmo_id"], "cycle": sample_float["cycles"][-1]}
    data = client.post("/api/forecast", json=payload).json()
    assert data["has_observation"] is False
    assert data["metrics_comparison"] == []
    assert data["profiles"][0]["observed_temperature"] is None


def test_forecast_rejects_short_history(client, sample_float):
    payload = {"wmoId": sample_float["wmo_id"], "cycle": sample_float["cycles"][0]}
    res = client.post("/api/forecast", json=payload)
    assert res.status_code == 400
    assert "Select cycle" in res.json()["detail"]


def test_chat_endpoint_grounded(client, sample_float):
    payload = {"query": "What is the temperature at 100 dbar?",
               "wmoId": sample_float["wmo_id"], "cycle": sample_float["defaultCycle"]}
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["verified"] is True
    assert len(data["citations"]) >= 1
    assert sample_float["wmo_id"] in data["text"]


def test_model_info_endpoint(client):
    meta = client.get("/api/model").json()
    assert meta["model_name"] == "ArgoCast PI-BiLSTM"
    assert meta["test_rmse_temp"] < meta["test_rmse_temp_persistence"]


def test_benchmark_endpoint(client):
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    b = res.json()
    assert len(b["models"]) == 13
    for split in ("chronological", "unseen_floats"):
        rows = b["splits"][split]["rows"]
        assert set(rows) == {m["name"] for m in b["models"]}
        fc = rows["ArgoCast PI-BiLSTM"]
        assert fc["temp_rmse"] < rows["Persistence"]["temp_rmse"]
        assert fc["inv_003_pct"] < 0.1
