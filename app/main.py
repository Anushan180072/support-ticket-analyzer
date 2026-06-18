"""
Run with:  uvicorn app.main:app --reload
Then visit http://localhost:8000/        for the minimal UI
        or http://localhost:8000/docs    for interactive API docs

Endpoints:
    GET  /api/health           - service + data + LLM status
    POST /api/query            - natural language question -> answer
    GET  /api/anomalies        - run anomaly detectors, return findings
    GET  /api/tickets          - simple filtered ticket listing (bonus, used by the UI)
    GET  /api/summary          - quick dataset stats (bonus, used by the UI dashboard)
    GET  /                     - minimal HTML UI
"""
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.ingestion import load_dataframe, build_sqlite
from app.llm_client import llm_client
from app.nl_query import NLQueryEngine
from app.anomalies import run_all_detectors, to_dicts, DETECTORS
from app.schemas import (
    QueryRequest, QueryResponse, AnomalyResponse, AnomalyItem, HealthResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ticket_ai.main")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading dataset from {settings.DATA_PATH}")
    df, report = load_dataframe(settings.DATA_PATH)
    conn = build_sqlite(df, settings.DB_PATH)
    conn.execute("PRAGMA query_only = ON")

    state["df"] = df
    state["conn"] = conn
    state["report"] = report
    state["engine"] = NLQueryEngine(conn, df)

    logger.info(
        f"Ready: {report.rows_loaded} rows loaded, {report.rows_dropped} dropped, "
        f"LLM provider = {llm_client.provider}"
    )
    yield
    conn.close()


app = FastAPI(
    title="Support Ticket AI",
    description="NL query + anomaly detection over a customer support ticket dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health():
    report = state["report"]
    return HealthResponse(
        status="ok",
        rows_loaded=report.rows_loaded,
        rows_dropped=report.rows_dropped,
        data_warnings=report.warnings,
        llm_provider=llm_client.provider,
        llm_available=llm_client.available,
    )


@app.post("/api/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest):
    try:
        result = state["engine"].answer(req.question)
    except Exception as e:
        logger.exception("Unhandled error answering query")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}") from e
    return QueryResponse(
        answer=result.answer, sql=result.sql, columns=result.columns,
        rows=result.rows, engine=result.engine, warnings=result.warnings,
    )


@app.get("/api/anomalies", response_model=AnomalyResponse, tags=["anomalies"])
def anomalies(
    types: list[str] | None = Query(
        default=None,
        description=f"Subset of detector types to run. Available: {list(DETECTORS.keys())}",
    ),
    severity: str | None = Query(default=None, description="Filter to 'high', 'medium', or 'low'."),
):
    if types:
        unknown = [t for t in types if t not in DETECTORS]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown anomaly type(s): {unknown}. Available: {list(DETECTORS.keys())}",
            )
    found = run_all_detectors(state["df"], types)
    if severity:
        found = [a for a in found if a.severity == severity]
    by_severity: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for a in found:
        by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        by_type[a.type] = by_type.get(a.type, 0) + 1
    return AnomalyResponse(
        total=len(found),
        by_severity=by_severity,
        by_type=by_type,
        anomalies=[AnomalyItem(**d) for d in to_dicts(found)],
    )


@app.get("/api/tickets", tags=["data"])
def list_tickets(
    status: str | None = None,
    priority: str | None = None,
    category: str | None = None,
    agent_id: str | None = None,
    limit: int = Query(default=50, le=500),
):
    df = state["df"]
    out = df
    if status:
        out = out[out["status"] == status]
    if priority:
        out = out[out["priority"] == priority]
    if category:
        out = out[out["category"] == category]
    if agent_id:
        out = out[out["agent_id"] == agent_id]
    out = out.head(limit).copy()
    out["created_at"] = out["created_at"].astype(str)
    out = out.astype(object).where(pd.notnull(out), None)
    return out.to_dict(orient="records")


@app.get("/api/summary", tags=["data"])
def summary():
    df = state["df"]
    return {
        "total_tickets": len(df),
        "by_status": df["status"].value_counts().to_dict(),
        "by_priority": df["priority"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "avg_resolution_time_hrs": round(df["resolution_time_hrs"].dropna().mean(), 2)
            if df["resolution_time_hrs"].notna().any() else None,
        "avg_response_time_hrs": round(df["response_time_hrs"].dropna().mean(), 2)
            if df["response_time_hrs"].notna().any() else None,
        "avg_customer_rating": round(df["customer_rating"].dropna().mean(), 2)
            if df["customer_rating"].notna().any() else None,
    }


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", tags=["ui"])
def root():
    return FileResponse("app/static/index.html")
