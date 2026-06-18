import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from app.ingestion import SCHEMA_DESCRIPTION
from app.llm_client import llm_client, LLMUnavailableError

logger = logging.getLogger("ticket_ai.nl_query")

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|REPLACE|CREATE|TRUNCATE|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)

SQL_SYSTEM_PROMPT = f"""You are a SQLite expert helping answer questions about a customer \
support ticket dataset.

{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY a JSON object: {{"sql": "<single SELECT statement>", "explanation": "<one short sentence>"}}
- The SQL must be a single read-only SELECT statement (CTEs with WITH...SELECT are fine).
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, or any statement that modifies data or schema.
- Do not include a trailing semicolon if it's not needed; do not include multiple statements.
- If the question cannot be answered from the `tickets` table, return:
  {{"sql": null, "explanation": "why it can't be answered"}}
- Use julianday('now') - julianday(created_at) for ticket age in days when needed (multiply by 24 for hours).
- Always return valid JSON and nothing else -- no markdown fences, no commentary outside the JSON.
"""

ANSWER_SYSTEM_PROMPT = """You are a precise data analyst. You will be given a user's question, \
the SQL query that was run, and the resulting rows. Write a short, direct, natural-language \
answer (1-3 sentences) using ONLY the numbers/values present in the results. Do not invent \
numbers. where ever required give emojis also. If the result set is empty, say so plainly. Do not mention SQL or databases in your answer.
"""


@dataclass
class QueryResult:
    answer: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    engine: str = "llm"
    warnings: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _validate_sql(sql: str) -> str:
    if sql is None:
        raise ValueError("empty")
    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"^\s*(WITH|SELECT)\b", stripped, re.IGNORECASE):
        raise ValueError("Generated SQL must start with SELECT or WITH.")
    if FORBIDDEN_KEYWORDS.search(stripped):
        raise ValueError("Generated SQL contains a forbidden (write/DDL) keyword.")
    if ";" in stripped:
        raise ValueError("Multiple SQL statements are not allowed.")
    return stripped


class NLQueryEngine:
    def __init__(self, conn: sqlite3.Connection, df: pd.DataFrame):
        self.conn = conn
        self.df = df

    def answer(self, question: str) -> QueryResult:
        if not question or not question.strip():
            return QueryResult(answer="Please ask a non-empty question.", engine="validation")

        if llm_client.available:
            try:
                return self._answer_via_llm(question)
            except LLMUnavailableError as e:
                logger.warning(f"LLM path failed, falling back to rule-based engine: {e}")
                result = self._answer_via_fallback(question)
                result.warnings.append(f"LLM unavailable ({e}); used rule-based fallback.")
                return result
        else:
            result = self._answer_via_fallback(question)
            result.warnings.append(
                "No LLM provider configured (set GROQ_API_KEY); used rule-based fallback."
            )
            return result

    def _generate_sql(self, question: str, prior_error: str | None = None) -> tuple[str | None, str]:
        user_prompt = f"Question: {question}"
        if prior_error:
            user_prompt += (
                f"\n\nYour previous SQL failed with this error:\n{prior_error}\n"
                f"Fix the SQL and try again."
            )
        raw = llm_client.chat(SQL_SYSTEM_PROMPT, user_prompt, json_mode=True)
        try:
            parsed = _extract_json(raw)
        except json.JSONDecodeError as e:
            raise LLMUnavailableError(f"Model did not return valid JSON for SQL: {e}") from e
        return parsed.get("sql"), parsed.get("explanation", "")

    def _answer_via_llm(self, question: str) -> QueryResult:
        sql, explanation = self._generate_sql(question)

        if sql is None:
            return QueryResult(
                answer=f"I can't answer that from this dataset. {explanation}".strip(),
                engine="llm",
            )

        try:
            sql = _validate_sql(sql)
        except ValueError as e:
            # one repair attempt, feeding the validation error back in
            sql2, _ = self._generate_sql(question, prior_error=str(e))
            sql = _validate_sql(sql2)

        try:
            cols, rows = self._execute(sql)
        except sqlite3.Error as e:
            # one repair attempt for execution errors too
            sql2, _ = self._generate_sql(question, prior_error=str(e))
            sql2 = _validate_sql(sql2)
            cols, rows = self._execute(sql2)
            sql = sql2

        nl_answer = self._summarize(question, sql, cols, rows)
        return QueryResult(answer=nl_answer, sql=sql, columns=cols, rows=rows, engine="llm")

    def _execute(self, sql: str) -> tuple[list[str], list[dict]]:
        cur = self.conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(200)]
        return cols, rows

    def _summarize(self, question: str, sql: str, cols: list[str], rows: list[dict]) -> str:
        payload = {
            "question": question,
            "sql": sql,
            "row_count": len(rows),
            "rows_sample": rows[:25],
        }
        try:
            return llm_client.chat(
                ANSWER_SYSTEM_PROMPT, json.dumps(payload, default=str)
            ).strip()
        except LLMUnavailableError:
            return self._template_answer(cols, rows)

    @staticmethod
    def _template_answer(cols: list[str], rows: list[dict]) -> str:
        if not rows:
            return "No matching rows were found."
        if len(rows) == 1 and len(cols) == 1:
            return f"{cols[0]}: {rows[0][cols[0]]}"
        return f"Found {len(rows)} result(s): {rows[:5]}"

    #deterministic fallback (no LLM available)
    def _answer_via_fallback(self, question: str) -> QueryResult:
        from app.fallback_nl import answer_with_rules
        return answer_with_rules(self.df, question)
