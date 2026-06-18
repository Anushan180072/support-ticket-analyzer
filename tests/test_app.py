import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.ingestion import load_dataframe, build_sqlite
from app.anomalies import run_all_detectors
from app.fallback_nl import answer_with_rules
from app.nl_query import NLQueryEngine
import app.llm_client as llm_client_module


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "support_tickets.csv")


@pytest.fixture(scope="module")
def df():
    d, _ = load_dataframe(DATA_PATH)
    return d


@pytest.fixture(scope="module")
def conn(df):
    return build_sqlite(df)


# ---------------- ingestion ----------------

def test_ingestion_loads_500_rows(df):
    assert len(df) == 500
    assert set(df["priority"].unique()) <= {"Low", "Medium", "High", "Critical"}
    assert set(df["status"].unique()) <= {"Open", "Resolved", "Escalated"}


# ---------------- anomalies ----------------

def test_anomalies_detect_something(df):
    anomalies = run_all_detectors(df)
    assert len(anomalies) > 0
    types_found = {a.type for a in anomalies}
    assert "stale_unresolved_high_priority" in types_found


def test_anomalies_filterable_by_type(df):
    anomalies = run_all_detectors(df, types=["resolution_time_outlier"])
    assert all(a.type == "resolution_time_outlier" for a in anomalies)


# ---------------- fallback NL engine ----------------

def test_fallback_unresolved_does_not_match_resolved_substring(df):
    """Regression test for a real bug found during development: 'unresolved'
    contains the substring 'resolved', which incorrectly matched the
    Resolved status filter and zeroed out every result."""
    result = answer_with_rules(df, "How many critical tickets are unresolved?")
    expected = len(df[(df["priority"] == "Critical") & (df["status"] != "Resolved")])
    assert str(expected) in result.answer
    assert expected > 0


def test_fallback_agent_rating_question(df):
    result = answer_with_rules(df, "Which agent has the lowest average customer rating?")
    assert "AGT-" in result.answer


def test_fallback_breakdown_question(df):
    result = answer_with_rules(df, "Give me a breakdown by priority")
    assert result.rows


# ---------------- LLM-path NL engine (mocked) ----------------

class _FakeLLM:
    """Stands in for llm_client so the SQL pipeline can be tested without
    network access or an API key. `script` is a list of canned responses
    returned in order across successive .chat() calls."""
    def __init__(self, script):
        self.script = list(script)
        self.available = True
        self.provider = "fake"
        self.calls = []

    def chat(self, system, user, json_mode=False, temperature=0.0):
        self.calls.append(user)
        if not self.script:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.script.pop(0)


def test_llm_path_happy_case(monkeypatch, conn, df):
    fake = _FakeLLM([
        json.dumps({"sql": "SELECT COUNT(*) as n FROM tickets WHERE priority='Critical' AND status != 'Resolved'",
                    "explanation": "count unresolved critical tickets"}),
        "There are 20 unresolved critical tickets.",
    ])
    monkeypatch.setattr("app.nl_query.llm_client", fake)
    engine = NLQueryEngine(conn, df)
    result = engine.answer("How many critical tickets are unresolved?")
    assert result.engine == "llm"
    assert result.sql is not None
    assert "20" in result.answer


def test_llm_path_rejects_malicious_sql_and_repairs(monkeypatch, conn, df):
    """If the model tries to emit a destructive statement, validation must
    reject it client-side (the DB connection is also read-only as a second
    layer of defense) and trigger exactly one repair attempt."""
    fake = _FakeLLM([
        json.dumps({"sql": "DROP TABLE tickets", "explanation": "oops"}),
        json.dumps({"sql": "SELECT COUNT(*) as n FROM tickets", "explanation": "fixed"}),
        "There are 500 tickets total.",
    ])
    monkeypatch.setattr("app.nl_query.llm_client", fake)
    engine = NLQueryEngine(conn, df)
    result = engine.answer("how many tickets are there")
    assert result.engine == "llm"
    assert "DROP" not in (result.sql or "")
    assert "500" in result.answer


def test_llm_path_unanswerable_question(monkeypatch, conn, df):
    fake = _FakeLLM([
        json.dumps({"sql": None, "explanation": "The dataset has no weather data."}),
    ])
    monkeypatch.setattr("app.nl_query.llm_client", fake)
    engine = NLQueryEngine(conn, df)
    result = engine.answer("What's the weather like today?")
    assert "weather" in result.answer.lower() or "can't answer" in result.answer.lower()


def test_falls_back_when_llm_unavailable(monkeypatch, conn, df):
    class _Unavailable:
        available = False
        provider = "fallback"
    monkeypatch.setattr("app.nl_query.llm_client", _Unavailable())
    engine = NLQueryEngine(conn, df)
    result = engine.answer("How many tickets are there in total?")
    assert result.engine == "fallback"


# ---------------- REST API ----------------

@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["rows_loaded"] == 500
    assert "llm_provider" in body


def test_query_endpoint(client):
    r = client.post("/api/query", json={"question": "How many tickets are there in total?"})
    assert r.status_code == 200
    assert "500" in r.json()["answer"]


def test_query_endpoint_rejects_empty_question(client):
    r = client.post("/api/query", json={"question": ""})
    assert r.status_code == 422  # pydantic min_length validation


def test_anomalies_endpoint(client):
    r = client.get("/api/anomalies")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(body["anomalies"])


def test_anomalies_endpoint_rejects_unknown_type(client):
    r = client.get("/api/anomalies", params={"types": "not_a_real_type"})
    assert r.status_code == 400


def test_tickets_endpoint_filters(client):
    r = client.get("/api/tickets", params={"priority": "Critical", "limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert all(row["priority"] == "Critical" for row in rows)


def test_root_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Support Ticket AI" in r.text
