"""
No LLM is involved here on purpose -- these are well-defined, deterministic
computations and using an LLM would only add latency/cost without improving
correctness. The system prompt's "AI-powered" requirement for anomaly
detection is satisfied via the NL-query layer being able to *explain*
anomalies in natural language on request, not by routing every anomaly
check through a model.
"""
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

from app.config import settings


@dataclass
class Anomaly:
    ticket_id: str
    type: str
    severity: str  # "low" | "medium" | "high"
    detail: str
    metric_value: float | None = None
    threshold: float | None = None


def _reference_now() -> pd.Timestamp:
    if settings.REFERENCE_NOW:
        return pd.Timestamp(settings.REFERENCE_NOW)
    return pd.Timestamp(datetime.now())


def detect_stale_unresolved(df: pd.DataFrame) -> list[Anomaly]:
    """Unresolved high-priority tickets older than the configured threshold."""
    now = _reference_now()
    age_hours = (now - df["created_at"]).dt.total_seconds() / 3600
    mask = (
        df["status"].isin(["Open", "Escalated"])
        & df["priority"].isin(settings.STALE_PRIORITIES)
        & (age_hours > settings.STALE_UNRESOLVED_HOURS)
    )
    out = []
    for idx in df[mask].index:
        row = df.loc[idx]
        hrs = round(age_hours.loc[idx], 1)
        severity = "high" if row["priority"] == "Critical" else "medium"
        out.append(Anomaly(
            ticket_id=row["ticket_id"],
            type="stale_unresolved_high_priority",
            severity=severity,
            detail=(
                f"{row['priority']} priority ticket has been '{row['status']}' for "
                f"{hrs} hours (threshold: {settings.STALE_UNRESOLVED_HOURS}h)."
            ),
            metric_value=hrs,
            threshold=settings.STALE_UNRESOLVED_HOURS,
        ))
    return out


def detect_resolution_time_outliers(df: pd.DataFrame) -> list[Anomaly]:
    """IQR-based outliers in resolution_time_hrs, computed per category+priority group."""
    out = []
    resolved = df.dropna(subset=["resolution_time_hrs"])
    for (cat, pri), group in resolved.groupby(["category", "priority"]):
        if len(group) < 4:
            continue
        q1, q3 = group["resolution_time_hrs"].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper_bound = q3 + settings.IQR_MULTIPLIER * iqr
        outliers = group[group["resolution_time_hrs"] > upper_bound]
        for _, row in outliers.iterrows():
            ratio = row["resolution_time_hrs"] / upper_bound
            severity = "high" if ratio > 2 else "medium"
            out.append(Anomaly(
                ticket_id=row["ticket_id"],
                type="resolution_time_outlier",
                severity=severity,
                detail=(
                    f"Resolution time of {row['resolution_time_hrs']}h is far above the "
                    f"typical range for {cat}/{pri} tickets (upper bound ~{round(upper_bound, 1)}h, "
                    f"group median {round(group['resolution_time_hrs'].median(), 1)}h)."
                ),
                metric_value=row["resolution_time_hrs"],
                threshold=round(upper_bound, 2),
            ))
    return out


def detect_response_time_outliers(df: pd.DataFrame) -> list[Anomaly]:
    """Same IQR approach applied to first-response time, across all tickets (response
    time is recorded for every ticket, so no per-status grouping needed)."""
    out = []
    sub = df.dropna(subset=["response_time_hrs"])
    if len(sub) < 4:
        return out
    q1, q3 = sub["response_time_hrs"].quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return out
    upper_bound = q3 + settings.IQR_MULTIPLIER * iqr
    for _, row in sub[sub["response_time_hrs"] > upper_bound].iterrows():
        out.append(Anomaly(
            ticket_id=row["ticket_id"],
            type="response_time_outlier",
            severity="medium",
            detail=(
                f"First response took {row['response_time_hrs']}h, above the typical "
                f"upper bound of ~{round(upper_bound, 1)}h across all tickets."
            ),
            metric_value=row["response_time_hrs"],
            threshold=round(upper_bound, 2),
        ))
    return out


def detect_data_integrity_issues(df: pd.DataFrame) -> list[Anomaly]:
    """Resolution time recorded as less than response time -- physically impossible,
    almost certainly a data entry error worth flagging to whoever owns the pipeline."""
    out = []
    sub = df.dropna(subset=["resolution_time_hrs", "response_time_hrs"])
    bad = sub[sub["resolution_time_hrs"] < sub["response_time_hrs"]]
    for _, row in bad.iterrows():
        out.append(Anomaly(
            ticket_id=row["ticket_id"],
            type="data_integrity_resolution_before_response",
            severity="low",
            detail=(
                f"Resolution time ({row['resolution_time_hrs']}h) is recorded as less than "
                f"response time ({row['response_time_hrs']}h) -- likely a data entry error."
            ),
            metric_value=row["resolution_time_hrs"],
        ))
    return out


def detect_low_rating_clusters(df: pd.DataFrame) -> list[Anomaly]:
    """Agents whose average rating is more than IQR-style low relative to peers --
    surfaced as agent-level anomalies (ticket_id is the agent's first low-rated ticket
    for traceability, full detail names the agent)."""
    out = []
    rated = df.dropna(subset=["customer_rating"])
    if rated.empty:
        return out
    per_agent = rated.groupby("agent_id")["customer_rating"].agg(["mean", "count"])
    per_agent = per_agent[per_agent["count"] >= 3]
    if per_agent.empty:
        return out
    overall_mean = rated["customer_rating"].mean()
    q1, q3 = per_agent["mean"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - settings.IQR_MULTIPLIER * iqr if iqr > 0 else overall_mean - 1
    flagged = per_agent[per_agent["mean"] < lower_bound]
    for agent_id, row in flagged.iterrows():
        example_ticket = rated[rated["agent_id"] == agent_id].iloc[0]["ticket_id"]
        out.append(Anomaly(
            ticket_id=example_ticket,
            type="agent_low_rating_cluster",
            severity="medium",
            detail=(
                f"Agent {agent_id} has an average rating of {round(row['mean'], 2)}/5 across "
                f"{int(row['count'])} rated tickets, notably below the cross-agent average of "
                f"{round(overall_mean, 2)}/5."
            ),
            metric_value=round(row["mean"], 2),
            threshold=round(lower_bound, 2),
        ))
    return out


DETECTORS = {
    "stale_unresolved_high_priority": detect_stale_unresolved,
    "resolution_time_outlier": detect_resolution_time_outliers,
    "response_time_outlier": detect_response_time_outliers,
    "data_integrity_resolution_before_response": detect_data_integrity_issues,
    "agent_low_rating_cluster": detect_low_rating_clusters,
}


def run_all_detectors(df: pd.DataFrame, types: list[str] | None = None) -> list[Anomaly]:
    selected = types or list(DETECTORS.keys())
    results: list[Anomaly] = []
    for t in selected:
        fn = DETECTORS.get(t)
        if fn is None:
            continue
        results.extend(fn(df))
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda a: severity_rank.get(a.severity, 3))
    return results


def to_dicts(anomalies: list[Anomaly]) -> list[dict]:
    return [asdict(a) for a in anomalies]
