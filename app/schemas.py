from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, examples=[
        "How many critical tickets are unresolved?"
    ])


class QueryResponse(BaseModel):
    answer: str
    sql: str | None = None
    columns: list[str] = []
    rows: list[dict] = []
    engine: str
    warnings: list[str] = []


class AnomalyItem(BaseModel):
    ticket_id: str
    type: str
    severity: str
    detail: str
    metric_value: float | None = None
    threshold: float | None = None


class AnomalyResponse(BaseModel):
    total: int
    by_severity: dict[str, int]
    by_type: dict[str, int]
    anomalies: list[AnomalyItem]


class HealthResponse(BaseModel):
    status: str
    rows_loaded: int
    rows_dropped: int
    data_warnings: list[str]
    llm_provider: str
    llm_available: bool
