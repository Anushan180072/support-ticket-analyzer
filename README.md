# Support Ticket AI

Support Ticket AI is a small FastAPI application for exploring a customer-support ticket dataset using natural-language questions, anomaly detection, and a simple web UI.

The app loads a CSV dataset, normalizes the columns, stores the cleaned data in an in-memory SQLite database, and lets users ask questions such as:

- "How many critical tickets are unresolved?"
- "Which agent has the lowest average customer rating?"
- "What is the average resolution time for Technical tickets?"
- "Show a breakdown by priority"

## Features

- Natural-language query support using a text-to-SQL flow
- Deterministic fallback rules when no LLM is available
- Anomaly detection for stale unresolved tickets and outliers
- REST API endpoints for health checks, querying, ticket listing, and summaries
- Minimal browser UI served from the app

## Project structure

- `app/main.py` — FastAPI app and API routes
- `app/config.py` — environment/config settings
- `app/ingestion.py` — CSV loading, cleaning, schema normalization, SQLite creation
- `app/nl_query.py` — LLM-based natural-language query engine
- `app/fallback_nl.py` — rule-based fallback engine
- `app/anomalies.py` — anomaly detection logic
- `app/schemas.py` — Pydantic response/request models
- `app/static/index.html` — minimal frontend UI
- `data/support_tickets.csv` — sample dataset
- `tests/test_app.py` — pytest suite

## Requirements

- Python 3.10+
- pip
- A working internet connection if you want to use the Groq API

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The app reads settings from `.env` if present.

Important variables:

- `DATA_PATH` — path to the CSV file (default: `data/support_tickets.csv`)
- `DB_PATH` — SQLite database path (default: `:memory:`)
- `LLM_PROVIDER` — `auto` or `groq`
- `GROQ_API_KEY` — API key for Groq (optional if using fallback only)
- `FORCE_FALLBACK` — set to `true` to disable the LLM path and use fallback rules only
- `REFERENCE_NOW` — optional fixed date used to make anomaly checks stable for sample data

Example `.env`:

```env
DATA_PATH=data/support_tickets.csv
LLM_PROVIDER=auto
GROQ_API_KEY=your_api_key_here
REFERENCE_NOW=2024-03-15 12:00
```

## Running the app

Start the server:

```bash
uvicorn app.main:app --reload
```

Then open:

- http://localhost:8000/ — UI
- http://localhost:8000/docs — Swagger docs

## API endpoints

### Health check

```http
GET /api/health
```

Returns dataset stats and LLM availability.

### Ask a question

```http
POST /api/query
Content-Type: application/json

{
  "question": "How many critical tickets are unresolved?"
}
```

Returns:

- `answer`
- `sql` (if available)
- `columns`
- `rows`
- `engine`
- `warnings`

### Get anomalies

```http
GET /api/anomalies
```

Optional query params:

- `types=resolution_time_outlier`
- `severity=high`

### List tickets

```http
GET /api/tickets?priority=Critical&limit=5
```

### Dataset summary

```http
GET /api/summary
```

## How the question-answering flow works

1. The app loads and cleans the CSV.
2. The cleaned data is loaded into SQLite.
3. The question engine asks the LLM to produce SQL.
4. The SQL is validated to ensure it is read-only.
5. The SQL is executed against the SQLite database.
6. The LLM summarizes the result into a natural-language answer.
7. If the LLM is unavailable, a fallback rule engine handles common question patterns.

## Notes about the dataset

The project expects a ticket-like dataset with columns such as:

- `ticket_id`
- `created_at`
- `category`
- `priority`
- `status`
- `response_time_hrs`
- `resolution_time_hrs`
- `agent_id`
- `customer_rating`
- `issue_summary`

If you use a different CSV, it should follow the same schema (or use compatible aliases supported by the ingestion logic).

## Running tests

```bash
pytest -q
```

The tests are useful for confirming the behavior of ingestion, API endpoints, anomaly detection, and query handling.
