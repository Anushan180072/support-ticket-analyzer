import sqlite3
import logging
from dataclasses import dataclass, field

import pandas as pd

from app.config import settings

logger = logging.getLogger("ticket_ai.ingestion")

COLUMN_ALIASES = {
    "ticket_id": ["ticket_id", "id"],
    "created_at": ["created_at", "created", "timestamp"],
    "category": ["category"],
    "priority": ["priority"],
    "status": ["status"],
    "response_time_hrs": ["resp_time_hrs", "response_time_hrs", "response_time"],
    "resolution_time_hrs": ["resol_time_hrs", "resolution_time_hrs", "resolution_time"],
    "agent_id": ["agent_id", "agent"],
    "customer_rating": ["cust_rating", "customer_rating", "rating"],
    "issue_summary": ["issue_summary", "summary", "description"],
}

REQUIRED_COLUMNS = [
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary",
]

VALID_PRIORITIES = {"Low", "Medium", "High", "Critical"}
VALID_STATUSES = {"Open", "Resolved", "Escalated"}


@dataclass
class IngestionReport:
    rows_loaded: int = 0
    rows_dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever columns are present to the canonical names."""
    rename_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                rename_map[lower_cols[alias]] = canonical
                break
    df = df.rename(columns=rename_map)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns (after alias resolution): {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    return df[REQUIRED_COLUMNS]


def load_dataframe(csv_path: str) -> tuple[pd.DataFrame, IngestionReport]:
    report = IngestionReport()
    try:
        raw = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find dataset at '{csv_path}'. Set DATA_PATH in .env "
            f"or place the file at that path."
        ) from e
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV failed to parse: {e}") from e

    df = _resolve_columns(raw)
    n_before = len(df)

    dup_mask = df["ticket_id"].duplicated(keep="first")
    if dup_mask.any():
        report.warnings.append(f"Dropped {dup_mask.sum()} duplicate ticket_id rows.")
        df = df[~dup_mask]

    null_id_mask = df["ticket_id"].isna()
    if null_id_mask.any():
        report.warnings.append(f"Dropped {null_id_mask.sum()} rows with missing ticket_id.")
        df = df[~null_id_mask]

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    bad_dates = df["created_at"].isna()
    if bad_dates.any():
        report.warnings.append(f"Dropped {bad_dates.sum()} rows with unparseable created_at.")
        df = df[~bad_dates]

    for col in ("response_time_hrs", "resolution_time_hrs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")
    # Ratings should be 1-5 ints when present; clamp out-of-range values to NaN
    # rather than silently trusting bad data.
    invalid_rating = df["customer_rating"].notna() & (
        (df["customer_rating"] < 1) | (df["customer_rating"] > 5)
    )
    if invalid_rating.any():
        report.warnings.append(
            f"Found {invalid_rating.sum()} customer_rating values outside 1-5; set to null."
        )
        df.loc[invalid_rating, "customer_rating"] = pd.NA

    bad_priority = ~df["priority"].isin(VALID_PRIORITIES)
    if bad_priority.any():
        report.warnings.append(
            f"{bad_priority.sum()} rows have a priority outside {sorted(VALID_PRIORITIES)}: "
            f"{sorted(df.loc[bad_priority, 'priority'].unique().tolist())}"
        )
    bad_status = ~df["status"].isin(VALID_STATUSES)
    if bad_status.any():
        report.warnings.append(
            f"{bad_status.sum()} rows have a status outside {sorted(VALID_STATUSES)}: "
            f"{sorted(df.loc[bad_status, 'status'].unique().tolist())}"
        )

    df["issue_summary"] = df["issue_summary"].fillna("")
    df = df.reset_index(drop=True)

    report.rows_loaded = len(df)
    report.rows_dropped = n_before - len(df)
    for w in report.warnings:
        logger.warning(w)

    return df, report


def build_sqlite(df: pd.DataFrame, db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Load the cleaned DataFrame into SQLite so the LLM can write normal SQL
    against a `tickets` table. check_same_thread=False because FastAPI may
    serve requests from different threads in the dev server.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    write_df = df.copy()
    write_df["created_at"] = write_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    write_df.to_sql("tickets", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tickets(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON tickets(priority)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON tickets(agent_id)")
    conn.commit()
    return conn


SCHEMA_DESCRIPTION = """\
Table: tickets
  ticket_id           TEXT    -- unique ticket id, e.g. 'TKT-001'
  created_at          TEXT    -- 'YYYY-MM-DD HH:MM:SS', ticket creation time
  category            TEXT    -- one of 'Billing', 'Technical', 'General'
  priority            TEXT    -- one of 'Low', 'Medium', 'High', 'Critical'
  status              TEXT    -- one of 'Open', 'Resolved', 'Escalated'
  response_time_hrs   REAL    -- hours from creation to first agent response
  resolution_time_hrs REAL    -- hours from creation to resolution; NULL if unresolved
  agent_id            TEXT    -- e.g. 'AGT-04'
  customer_rating     INTEGER -- 1-5; NULL if unresolved
  issue_summary       TEXT    -- free-text description
Notes:
  - "unresolved" means status != 'Resolved' (i.e. 'Open' or 'Escalated').
  - Use SQLite date functions, e.g. strftime('%s', 'now') for current time,
    or julianday() for hour differences against created_at.
"""
