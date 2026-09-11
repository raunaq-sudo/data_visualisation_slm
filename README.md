# Dynamic SQL Dashboard Agent

A natural-language dashboard builder. Users chat with an AI agent (running locally via Ollama) to describe a widget they want — e.g. *"total revenue per product category"* — and the agent generates the SQL, runs it against a SQLite data store, and renders the result as a draggable/resizable chart widget on a canvas.

Built with **FastAPI**, **pydantic-ai**, and **Ollama** (`qwen2.5:7b`), backed by SQLite.

## Features

- Conversational widget configuration (KPI, BARCHART, PIECHART, LINECHART, STACKEDBAR, REPORT/TABLE)
- Natural language → SQL generation against a runtime-discovered schema
- Structured, server-validated session state with confirmation flow
- WebSocket chat with history persistence and replay on reconnect
- Dashboard / widget CRUD with canvas drag & resize persistence (bulk canvas save)
- Schema metadata browser and query execution endpoints
- Dark-themed, chart.js / hammer.js-based drag-and-drop canvas frontend

## Project Structure

```
.
├── app.py                        # FastAPI app, agents, WebSocket chat endpoint
├── agent_orchestrator.py         # Agent bootstrap / experiment scratch
├── config.py                     # Loads .env (OLLAMA_BASE_URL, MODEL)
├── system_prompts.py             # Builds the intake agent system prompt
├── database.py                   # Shared sqlite connection helper (used by app.py)
├── routers/
│   ├── database.py               # get_db() connection helper + path constants
│   ├── chat.py                   # Chat session list/get/delete endpoints
│   ├── dashboards.py             # Dashboard + widget CRUD, canvas save, query exec
│   └── metadata.py               # Read-only schema metadata endpoints
├── db_setup/
│   ├── db_setup.py               # Creates system tables + populates metadata
│   ├── data_db_setup.py          # Loads CSVs from db_setup/data/ into data.db
│   └── data/                     # Source CSVs (gitignored)
├── frontend.html                 # Main chat + canvas UI
├── dashboards.html               # Dashboard management UI
├── data_steward.html             # Data steward UI
└── test_ollama_connection.py     # Verifies connectivity to the local Ollama server
```

> Note: root-level `database.py`, `chat.py`, `dashboards.py` and `metadata.py` are legacy copies of the files under `routers/`; the active code lives in `routers/`.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a compatible model pulled, e.g.:
  ```bash
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (create a .env file with:)
# OLLAMA_BASE_URL=http://localhost:11434/v1
# MODEL=qwen2.5:7b-instruct-q4_K_M
```

## Database Setup

Place CSV files in `db_setup/data/`, then:

```bash
# Create data.db from the CSVs (table name = filename)
python db_setup/data_db_setup.py

# Create dashboard_system.db (system tables + metadata) and verify schema
python db_setup/db_setup.py
```

## Run

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open:
- http://127.0.0.1:8000/ — chat + widget canvas
- http://127.0.0.1:8000/dashboards.html — dashboards
- http://127.0.0.1:8000/data_steward.html — data steward
- http://127.0.0.1:8000/docs — OpenAPI docs

Verify the model connection first with `python test_ollama_connection.py`.

## How It Works

1. User connects to `WS /ws/chat/{chat_id}`.
2. The **intake agent** guides the conversation toward collecting `widget_type`, `query_description`, and confirmation. It outputs structured JSON (`IntakeOutput`) with a `completeness` score from 0.0 → 1.0.
3. Once confirmed, the **SQL agent** produces a validated SQLite `SELECT` via structured output (`SQLOutput`: SQL, confidence, possible data errors).
4. The SQL is executed against `db_setup/data.db` (with `LIMIT 100` applied only at fetch time), persisted to `widget_query_mapping`, and returned to the frontend in the WebSocket `system_status` payload.
5. The user places the widget on the canvas; geometry and chart config are persisted via the REST API.

The schema embedded in the prompts is discovered at startup from the `metadata_data_table` / `metadata_data_table_description` tables.

## API Overview

| Method | Path | Description |
| ------ | ---- | ----------- |
| WS | `/ws/chat/{chat_id}` | Conversational widget-building endpoint |
| GET | `/chat` | List chat sessions |
| GET/DELETE | `/chat/{chat_id}` | Fetch / delete a session |
| POST/GET/PATCH/DELETE | `/dashboards` | Dashboard CRUD |
| | `/dashboards/spaces` | Dashboards whose widgets are all approved |
| POST/GET/PATCH/DELETE | `/dashboards/{id}/widgets[/{uid}]` | Widget CRUD |
| PUT | `/dashboards/{id}/canvas` | Bulk-save widget geometry |
| POST | `/dashboards/{id}/widgets/{uid}/query` | Execute a widget's SQL |
| GET | `/metadata` | Table summaries |
| GET | `/metadata/tables[/{table}]` | Full schema metadata |