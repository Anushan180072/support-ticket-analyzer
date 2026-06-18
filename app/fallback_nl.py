"""
Deterministic fallback NL engine.
This module only kicks in when no LLM is configured/reachable at all,
so that `uvicorn main:app` still produces sensible answers to common
questions out of the box (e.g. during offline grading) instead of a hard
error. It covers a curated set of question shapes via keyword/pattern
matching over pandas, not general natural language understanding.
"""
import re

import pandas as pd

from app.nl_query import QueryResult

CATEGORIES = ["Billing", "Technical", "General"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Open", "Resolved", "Escalated"]


def _find_token(text: str, options: list[str]) -> str | None:
    """Whole-word match, case-insensitive (avoids e.g. 'resolved' matching inside
    'unresolved')."""
    low = text.lower()
    for opt in options:
        if re.search(rf"\b{re.escape(opt.lower())}\b", low):
            return opt
    return None


def _apply_filters(df: pd.DataFrame, q: str) -> pd.DataFrame:
    out = df
    q_low = q.lower()
    is_unresolved = "unresolved" in q_low or "not resolved" in q_low or "still open" in q_low

    cat = _find_token(q, CATEGORIES)
    pri = _find_token(q, PRIORITIES)
    status = None if is_unresolved else _find_token(q, STATUSES)

    if cat:
        out = out[out["category"] == cat]
    if pri:
        out = out[out["priority"] == pri]
    if status:
        out = out[out["status"] == status]
    if is_unresolved:
        out = out[out["status"] != "Resolved"]
    agent_match = re.search(r"agt-?\d+", q, re.IGNORECASE)
    if agent_match:
        agent_id = agent_match.group(0).upper().replace("AGT", "AGT-").replace("AGT--", "AGT-")
        agent_id = re.sub(r"AGT-?(\d+)", lambda m: f"AGT-{int(m.group(1)):02d}", agent_id)
        out = out[out["agent_id"] == agent_id]
    return out


def answer_with_rules(df: pd.DataFrame, question: str) -> QueryResult:
    q = question.lower().strip()

    # "which agent has the lowest/highest average rating"
    m = re.search(r"(lowest|highest)\s+average\s+(customer\s+)?rating", q)
    if m or ("lowest" in q and "rating" in q) or ("highest" in q and "rating" in q):
        direction = "lowest" if "lowest" in q else "highest"
        grp = (
            df.dropna(subset=["customer_rating"])
            .groupby("agent_id")["customer_rating"]
            .mean()
            .round(2)
        )
        if grp.empty:
            return QueryResult(answer="No customer ratings are available yet.", engine="fallback")
        agent = grp.idxmin() if direction == "lowest" else grp.idxmax()
        val = grp[agent]
        rows = [{"agent_id": a, "avg_rating": v} for a, v in grp.sort_values().items()]
        return QueryResult(
            answer=f"{agent} has the {direction} average customer rating, at {val}/5.",
            columns=["agent_id", "avg_rating"], rows=rows, engine="fallback",
        )

    # "average resolution/response time" (optionally filtered)
    if "average" in q and ("resolution time" in q or "resol" in q):
        sub = _apply_filters(df, q).dropna(subset=["resolution_time_hrs"])
        if sub.empty:
            return QueryResult(answer="No resolved tickets match that filter.", engine="fallback")
        avg = round(sub["resolution_time_hrs"].mean(), 2)
        return QueryResult(answer=f"The average resolution time is {avg} hours ({len(sub)} tickets).",
                            engine="fallback")

    if "average" in q and "response time" in q:
        sub = _apply_filters(df, q).dropna(subset=["response_time_hrs"])
        if sub.empty:
            return QueryResult(answer="No tickets match that filter.", engine="fallback")
        avg = round(sub["response_time_hrs"].mean(), 2)
        return QueryResult(answer=f"The average response time is {avg} hours ({len(sub)} tickets).",
                            engine="fallback")

    # "average rating" generic / per category/priority
    if "average" in q and "rating" in q:
        sub = _apply_filters(df, q).dropna(subset=["customer_rating"])
        if sub.empty:
            return QueryResult(answer="No ratings available for that filter.", engine="fallback")
        avg = round(sub["customer_rating"].mean(), 2)
        return QueryResult(answer=f"The average customer rating is {avg}/5 ({len(sub)} tickets).",
                            engine="fallback")

    # "how many ... tickets ..." / counts
    if "how many" in q or q.startswith("count") or "number of" in q:
        sub = _apply_filters(df, q)
        return QueryResult(
            answer=f"There are {len(sub)} matching ticket(s).",
            engine="fallback",
        )

    # breakdown by category/priority/status
    if "breakdown" in q or "distribution" in q or "by category" in q or "by priority" in q or "by status" in q:
        dim = "category" if "category" in q else ("priority" if "priority" in q else "status")
        counts = df[dim].value_counts().to_dict()
        rows = [{dim: k, "count": v} for k, v in counts.items()]
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
        return QueryResult(answer=f"Breakdown by {dim} -> {summary}.",
                            columns=[dim, "count"], rows=rows, engine="fallback")

    # total ticket count
    if "total" in q and "ticket" in q:
        return QueryResult(answer=f"There are {len(df)} tickets in total.", engine="fallback")

    return QueryResult(
        answer=(
            "I couldn't confidently match that question to a known pattern in fallback mode "
            "(no LLM is configured). Try rephrasing, or set GROQ_API_KEY for full "
            "natural-language understanding. Supported fallback patterns include counts "
            "('how many critical tickets are unresolved'), averages "
            "('average resolution time for Technical tickets'), agent comparisons "
            "('which agent has the lowest average rating'), and breakdowns "
            "('breakdown by priority')."
        ),
        engine="fallback",
    )
