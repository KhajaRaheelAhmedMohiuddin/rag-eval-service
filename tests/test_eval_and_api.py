"""Integration tests: the eval harness meets quality bars, and the API works."""
from fastapi.testclient import TestClient

from app.main import app
from eval.run_eval import run

client = TestClient(app)


def test_eval_meets_quality_bar():
    summary = run()
    # Deterministic offline pipeline should clear these bars on the golden set.
    assert summary["retrieval_hit_rate"] >= 0.85
    assert summary["answer_accuracy"] >= 0.70
    assert summary["refusal_accuracy"] >= 0.85


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["chunks_indexed"] > 0


def test_ask_returns_grounded_answer_with_sources():
    r = client.post("/ask", json={"question": "How much does the Premium plan cost?"})
    assert r.status_code == 200
    body = r.json()
    assert "9.99" in body["answer"]
    assert body["grounded"] is True
    assert len(body["sources"]) > 0


def test_ask_refuses_out_of_scope():
    r = client.post("/ask", json={"question": "Does AcmePay offer cryptocurrency trading?"})
    body = r.json()
    assert body["grounded"] is False
    assert body["needs_human_review"] is True


def test_ask_flags_high_stakes_for_hitl():
    r = client.post("/ask", json={"question": "How do I request a refund from a merchant?"})
    body = r.json()
    assert body["needs_human_review"] is True
