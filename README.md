# Dashboard Studio

An AI-powered conversational dashboard builder. Describe your data in plain English, and the system generates SQL queries, executes them against your database, and renders interactive dashboard widgets — all through a chat interface.

## How It Works

1. **Chat** with an AI agent via WebSocket to describe what you want to visualize
2. The agent collects your intent: widget type, data query, and confirmation
3. A second AI agent **generates a SQLite query** from your natural-language description
4. The query executes against your database and results are sent to the frontend
5. Results render as **interactive widgets** (KPI, bar chart, pie chart, line chart, stacked bar chart) on a draggable canvas

## Tech Stack

- **Backend:** Python, FastAPI, WebSockets
- **AI Agents:** PydanticAI with structured output (intake + SQL generation)
- **LLM:** Local Qwen 2.5 7B via [Ollama](https://ollama.com/)
- **Database:** SQLite
- **Frontend:** Vanilla HTML/CSS/JS, [Chart.js](https://www.chartjs.org/) 4.4

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- Pull the required model:
  ```bash
  ollama pull qwen2.5:7b-instruct
  ```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd hackathon
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env   # or create .env manually
   ```
   Edit `.env` with your Ollama base URL and model name:
   ```
   OLLAMA_BASE_URL=http://localhost:11434/v1
   MODEL=qwen2.5:7b-instruct
   ```

4. **Add your data:**
   Place CSV files in `db_setup/data/`, then run:
   ```bash
   python db_setup/data_db_setup.py
   ```

5. **Initialize the system database:**
   ```bash
   python db_setup/db_setup.py
   ```

6. **Start the server:**
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

7. **Open the app:**
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Project Structure

```
.
├── app.py                  # Main entry point — FastAPI app, WebSocket endpoint, AI agents
├── config.py               # Environment variable loading
├── database.py             # Shared SQLite connection helper
├── system_prompts.py       # LLM system prompt builder with schema embedding
├── frontend.html           # Dashboard builder UI (chat + canvas)
├── dashboards.html         # Dashboard listing/management page
├── data_steward.html       # Data steward / schema browser page
├── routers/
│   ├── chat.py             # Chat session endpoints
│   ├── dashboards.py       # Dashboard + widget CRUD
│   └── metadata.py         # Schema browser endpoints
├── db_setup/
│   ├── db_setup.py         # System database initialization
│   ├── data_db_setup.py    # CSV-to-SQLite ingestion
│   └── data/               # Place your CSV files here
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard builder UI |
| `GET` | `/dashboards.html` | Dashboard listing page |
| `GET` | `/data_steward.html` | Schema browser page |
| `WS` | `/ws/chat/{chat_id}` | WebSocket — conversational AI agent |
| `GET/POST/PATCH/DELETE` | `/dashboards` | Dashboard CRUD |
| `GET/POST/PATCH/DELETE` | `/dashboards/{id}/widgets` | Widget CRUD |
| `PUT` | `/dashboards/{id}/canvas` | Bulk-save widget positions |
| `GET` | `/metadata` | List tables with descriptions |
| `GET` | `/metadata/tables` | Full schema with columns |
| `GET/DELETE` | `/chat`, `/chat/{id}` | Chat session management |

## License

[MIT](LICENSE)
