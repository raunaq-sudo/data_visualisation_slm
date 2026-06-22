import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import List, Literal, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.output import NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
import config
# =====================================================================
# 0. SERVER LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DashboardAgent")

app = FastAPI(title="Dynamic SQL Dashboard Agent API")
DB_FILE_PATH = "db_setup/data.db"


# =====================================================================
# 1. DATABASE COMPONENT & CHAT LOGGING REPOSITORY
# =====================================================================

@contextmanager
def get_db_connection(db_path: str = DB_FILE_PATH):
    """Context manager ensuring DB connections are always closed, even on error."""
    connection = sqlite3.connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_chat_history_table():
    """Ensures the chat history table exists in our SQLite database."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
    logger.info("SQLite Chat history infrastructure initialized.")


def log_chat_message(chat_id: str, sender: str, message: str):
    """Inserts a structured interaction frame into the database table."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_message_history (chat_id, sender, message) VALUES (?, ?, ?);",
            (chat_id, sender, message)
        )
    logger.info(f"[{chat_id}] Transaction logged to DB -> {sender.upper()}: {message[:50]}...")


init_chat_history_table()


# =====================================================================
# 2. SCHEMA DISCOVERY FOR THE SQL AGENT
# =====================================================================

class ColumnSchema(BaseModel):
    name: str
    type: str


class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnSchema]


class DataSourceSchema(BaseModel):
    source_name: str
    tables: List[TableSchema]


class SQLAgentDeps:
    def __init__(self, schema: DataSourceSchema):
        self.schema = schema


# FIX: Table names are sanitized against a whitelist fetched from sqlite_master
# to prevent SQL injection via maliciously crafted table names.
def generate_registry_from_sqlite(db_path: str) -> DataSourceSchema:
    if not os.path.exists(db_path):
        logger.warning(f"Database target '{db_path}' not found at startup. Defaulting to empty schema layout.")
        return DataSourceSchema(source_name="empty_db", tables=[])

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        # Fetch trusted table names directly from sqlite_master
        trusted_table_names = [row[0] for row in cursor.fetchall()]

        table_schemas = []
        for table_name in trusted_table_names:
            # PRAGMA table_info accepts parameterized identifiers via cursor description;
            # since SQLite doesn't support ? placeholders in PRAGMA, we use the trusted
            # name fetched directly from sqlite_master (not from user input).
            cursor.execute(f"PRAGMA table_info(\"{table_name.replace('\"', '')}\");")
            columns_info = cursor.fetchall()
            column_schemas = [ColumnSchema(name=col[1], type=col[2]) for col in columns_info]
            table_schemas.append(TableSchema(table_name=table_name, columns=column_schemas))

    schema_res = DataSourceSchema(
        source_name=os.path.splitext(os.path.basename(db_path))[0],
        tables=table_schemas
    )
    logger.info(f"Database architecture auto-discovered. Tables found: {[t.table_name for t in schema_res.tables]}")
    return schema_res


db_schema = generate_registry_from_sqlite(DB_FILE_PATH)
dependencies = SQLAgentDeps(schema=db_schema)


# =====================================================================
# 3. STRUCTURED OUTPUT SCHEMA FOR INTAKE AGENT
# =====================================================================

ALLOWED_WIDGETS = {"KPI", "BARCHART", "PIECHART", "LINECHART", "STACKEDBAR"}


class IntakeOutput(BaseModel):
    """Structured output returned by the intake agent on every turn."""

    completeness: Literal[0.0, 0.3, 0.5, 0.7, 0.9, 1.0] = Field(
        description=(
            "How complete the user's data-visualisation request is. "
            "Pick exactly one of the allowed values: "
            "0.0 = greeting / off-topic / no useful info; "
            "0.3 = vague data interest but neither widget nor query is clear; "
            "0.5 = one of widget OR query is known, the other is still missing; "
            "0.7 = both widget AND query are known but user has not yet confirmed; "
            "0.9 = user has confirmed, finalising; "
            "1.0 = fully confirmed, is_confirmed must be True."
        ),
    )
    widget_type: Optional[str] = Field(
        default=None,
        description=f"One of {sorted(ALLOWED_WIDGETS)}, or null if not yet determined.",
    )
    query_description: Optional[str] = Field(
        default=None,
        description="The user's data query goal in plain English, or null if not yet clear.",
    )
    is_confirmed: bool = Field(
        default=False,
        description="True only when the user has explicitly agreed to proceed with the shown widget + query.",
    )
    reply: str = Field(
        description="Your next conversational message to the user.",
    )


# =====================================================================
# 4. AGENT PIPELINE DEFINITIONS
# =====================================================================
# Corrected profile for self-hosted Ollama v0.5+ with qwen2.5:7b-instruct.
# pydantic-ai's built-in qwen_model_profile does not set supports_json_schema_output
# for the 2.5 series, but self-hosted Ollama enforces the schema at the sampler
# level via llama.cpp grammar-constrained decoding, so we explicitly enable it.
# This switches the intake agent from prompt-based compliance (unreliable on 7B)
# to token-level schema enforcement (guaranteed valid JSON every time).
_qwen25_local_profile = ModelProfile(
    json_schema_transformer=InlineDefsJsonSchemaTransformer,
    ignore_streamed_leading_whitespace=True,
    supports_json_schema_output=True,
    supports_json_object_output=True,
)

local_qwen_model = OllamaModel(
    model_name='qwen2.5:7b-instruct',
    profile=_qwen25_local_profile,
)

_INTAKE_SYSTEM_PROMPT = """
You are a data visualisation assistant. Your only job is to help the user configure a dashboard widget by collecting three things: widget_type, query_description, and confirmation. You must always reply with a JSON object — no prose, no markdown, just raw JSON.

Allowed widget_type values: KPI, BARCHART, PIECHART, LINECHART, STACKEDBAR.

=== STRICT OUTPUT RULES ===
- Output ONLY a raw JSON object matching the schema below. No text before or after it.
- "is_confirmed" must be true ONLY when the user sends a clear affirmative reply (yes, confirm, ok, looks good, go ahead, sure, correct, that's right, yep, etc.) AFTER you have already shown them a summary. If the user has not yet seen a summary, is_confirmed must be false.
- "completeness" must be exactly one of: 0.0, 0.3, 0.5, 0.7, 0.9, 1.0.

=== completeness VALUES ===
0.0 → user sent a greeting, off-topic message, or nothing useful.
0.3 → user mentioned data vaguely but widget_type AND query_description are both still unknown.
0.5 → exactly one of widget_type OR query_description is now known; the other is missing.
0.7 → both widget_type AND query_description are filled; you have shown a summary; waiting for user confirmation.
0.9 → user just confirmed (said yes/ok/confirm/etc.) — set is_confirmed to true NOW in this response.
1.0 → is_confirmed is true and both fields are populated. Conversation is complete.

=== CONVERSATION STEPS (follow in order) ===
Step 1 — If widget_type is null: ask the user which widget type they want. List the 5 options.
Step 2 — If widget_type is set but query_description is null: ask what data/metric they want to display and any filters.
Step 3 — If both are set and is_confirmed is false: show a summary ("You want a BARCHART showing X. Shall I proceed?") and wait.
Step 4 — If user replies with any affirmative (yes, ok, sure, correct, go ahead, yep, looks good, confirm): set is_confirmed to true and completeness to 0.9. Thank the user.

=== OFF-TOPIC RULE ===
If the user says anything unrelated to data visualisation, do NOT answer it. Redirect them: "I can only help you build a dashboard widget. [continue with current step]". Set completeness to 0.0.

=== EXAMPLES ===

User: "hello"
{"completeness": 0.0, "widget_type": null, "query_description": null, "is_confirmed": false, "reply": "Hi! I'm here to help you build a data visualisation widget. Which widget type would you like — KPI, BARCHART, PIECHART, LINECHART, or STACKEDBAR?"}

User: "I want to see sales data"
{"completeness": 0.3, "widget_type": null, "query_description": null, "is_confirmed": false, "reply": "Great, sales data it is. Which widget type would you like — KPI, BARCHART, PIECHART, LINECHART, or STACKEDBAR?"}

User: "a bar chart"
{"completeness": 0.5, "widget_type": "BARCHART", "query_description": null, "is_confirmed": false, "reply": "Got it — a BARCHART. What data or metric do you want to display, and are there any filters like date range or category?"}

User: "total revenue per product category for last quarter"
{"completeness": 0.7, "widget_type": "BARCHART", "query_description": "total revenue per product category for last quarter", "is_confirmed": false, "reply": "Here's what I have: a BARCHART showing total revenue per product category for last quarter. Shall I proceed?"}

User: "yes"
{"completeness": 0.9, "widget_type": "BARCHART", "query_description": "total revenue per product category for last quarter", "is_confirmed": true, "reply": "Perfect! Generating your dashboard widget now."}

User: "what is the weather today"
{"completeness": 0.0, "widget_type": null, "query_description": null, "is_confirmed": false, "reply": "I can only help you build a dashboard widget. Which widget type would you like — KPI, BARCHART, PIECHART, LINECHART, or STACKEDBAR?"}
""".strip()

intake_agent = Agent(
    model=local_qwen_model,
    output_type=NativeOutput(IntakeOutput),   # schema enforced at sampler level
    system_prompt=_INTAKE_SYSTEM_PROMPT,
)

sql_query_agent = Agent(
    model=local_qwen_model,
    deps_type=SQLAgentDeps,
    retries=3,
    system_prompt="You are a strict SQL Generator. Output ONLY raw executable SQLite statement. Do not explain anything."
)


@sql_query_agent.system_prompt
def inject_database_schema(ctx: RunContext[SQLAgentDeps]) -> str:
    schema = ctx.deps.schema
    if not schema.tables:
        return "WARNING: No tables found in the database. You cannot generate a valid SQL query."
    instructions = f"Write SQLite statements for: '{schema.source_name}'.\nSchema:\n"
    for table in schema.tables:
        instructions += f"Table: {table.table_name}\n"
        for col in table.columns:
            instructions += f"  - {col.name} ({col.type})\n"
    return instructions


# =====================================================================
# 5. WEBSOCKET ROUTE WITH ROBUST PARSING
# =====================================================================

@app.websocket("/ws/chat/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    await websocket.accept()
    logger.info(f"WebSocket session open: [{chat_id}]")

    # FIX: message_history must accumulate ALL messages across turns.
    # Using result.all_messages() instead of result.new_messages() ensures
    # the agent retains full multi-turn context rather than only the last turn.
    message_history = []
    greeting_text = "Hello! Let's configure a dashboard widget. What kind of layout are we building? (e.g., KPI, BARCHART)"

    log_chat_message(chat_id=chat_id, sender="agent", message=greeting_text)
    await websocket.send_json({
        "sender": "agent",
        "message": greeting_text,
        "system_status": {"widget": None, "query": None, "confirmed": False, "sql": None}
    })

    try:
        while True:
            user_message = await websocket.receive_text()
            log_chat_message(chat_id=chat_id, sender="user", message=user_message)

            try:
                logger.info(f"[{chat_id}] Calling Intake Agent...")

                result = await intake_agent.run(
                    user_message,
                    message_history=message_history,
                    model_settings={'temperature': 0.1}
                )

                # Accumulate full conversation history for multi-turn context.
                message_history = result.all_messages()

                # result.output is now a validated IntakeOutput Pydantic instance.
                intake: IntakeOutput = result.output

                # Server-side confirmation safety net.
                # If the model failed to set is_confirmed=True despite a clear affirmative
                # from the user, and both fields are already populated, force it here.
                # This catches the most common small-model failure: correct prose reply
                # but wrong boolean in the JSON.
                _AFFIRMATIVES = {
                    "yes", "y", "yep", "yeah", "yup", "ok", "okay", "sure",
                    "confirm", "confirmed", "correct", "go ahead", "proceed",
                    "looks good", "that's right", "thats right", "do it",
                    "go for it", "absolutely", "affirmative", "agreed",
                }
                _user_lower = user_message.strip().lower().rstrip("!.,")
                _is_affirmative = (
                    _user_lower in _AFFIRMATIVES
                    or any(_user_lower.startswith(a) for a in _AFFIRMATIVES)
                )
                if (
                    _is_affirmative
                    and intake.widget_type is not None
                    and intake.query_description is not None
                    and not intake.is_confirmed
                ):
                    logger.warning(
                        f"[{chat_id}] Model missed confirmation — overriding is_confirmed=True "
                        f"(user said: {repr(user_message)})"
                    )
                    intake = intake.model_copy(update={
                        "is_confirmed": True,
                        "completeness": 0.9,
                    })

                logger.info(
                    f"[{chat_id}] Intake -> "
                    f"completeness={intake.completeness} | "
                    f"widget={intake.widget_type} | "
                    f"query={intake.query_description} | "
                    f"confirmed={intake.is_confirmed}"
                )

            except Exception as e:
                logger.error(f"[{chat_id}] Run Call Execution Failure!")
                logger.error(f"Error Type: {type(e).__name__} | Details: {e}")

                fallback_msg = "Processing issue encountered. Let's try that again."
                await websocket.send_json({
                    "sender": "agent",
                    "message": fallback_msg,
                    "system_status": {
                        "widget": None, "query": None,
                        "confirmed": False, "completeness": 0.0, "sql": None
                    }
                })
                continue

            generated_sql = None

            # Only proceed to SQL generation when the intake is sufficiently complete.
            # completeness > 0.8 means both widget + query are present and confirmed.
            if intake.completeness > 0.8 and intake.is_confirmed and intake.query_description:
                if not db_schema.tables:
                    generated_sql = "-- Error: No database tables found. Cannot generate SQL."
                    logger.warning(f"[{chat_id}] SQL generation skipped: empty schema.")
                else:
                    logger.info(f"[{chat_id}] completeness={intake.completeness:.2f} > 0.8 — generating SQL...")
                    try:
                        sql_result = await sql_query_agent.run(
                            f"Generate an accurate SQLite query for: {intake.query_description}",
                            deps=dependencies,
                            model_settings={'temperature': 0.1}
                        )
                        generated_sql = sql_result.output.strip()
                        logger.info(f"[{chat_id}] SQL generated successfully.")
                    except Exception as sql_err:
                        logger.error(f"SQL generation failed: {sql_err}")
                        generated_sql = f"-- Execution error: {sql_err}"

                # Reset conversation after a completed cycle (success or SQL error).
                message_history = []

            log_chat_message(chat_id=chat_id, sender="agent", message=intake.reply)

            await websocket.send_json({
                "sender": "agent",
                "message": intake.reply,
                "system_status": {
                    "widget": intake.widget_type,
                    "query": intake.query_description,
                    "confirmed": intake.is_confirmed,
                    "completeness": round(intake.completeness, 2),
                    "sql": generated_sql,
                }
            })

    except WebSocketDisconnect:
        logger.warning(f"Session closed for: {chat_id}")


# =====================================================================
# 6. FRONTEND VIEW
# =====================================================================
# FIX: The original frontend inserted data.message and data.system_status.query
# directly into innerHTML, which allows XSS if the model outputs <script> tags
# or HTML. All dynamic text now goes through textContent instead.
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Dashboard Studio</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/water.css">
    <style>
        #chat-box { height: 260px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background: #202b36; }
        .agent-msg { color: #4af; } .user-msg { color: #fff; }
        pre { background: #111; padding: 10px; border-radius: 4px; overflow-x: auto; color: #ffbc42; }
    </style>
</head>
<body>
    <h2>Live Dashboard Agent Hub</h2>
    <p>Active Session Room ID: <strong id="room-display" style="color:#4af;"></strong></p>
    <div id="chat-box"></div>
    <div>
        <input type="text" id="user-input" placeholder="Type to chat..." autocomplete="off" style="width:80%;">
        <button id="send-btn">Send</button>
    </div>
    <h3>System Slots</h3>
    <p>
      Widget: <span id="slot-widget" style="color:orange;">None</span> &nbsp;|&nbsp;
      Query: <span id="slot-query" style="color:orange;">Empty</span> &nbsp;|&nbsp;
      Completeness: <span id="slot-completeness" style="color:orange;">0.00</span>
      <span id="slot-completeness-gate" style="color:#888;font-size:0.85em;"></span>
    </p>
    <div style="background:#111;border-radius:4px;height:8px;width:100%;margin-bottom:12px;">
      <div id="completeness-bar" style="height:8px;border-radius:4px;width:0%;background:#888;transition:width 0.4s,background 0.4s;"></div>
    </div>
    <div id="sql-output-area" style="display:none;">
        <h3>Generated SQL:</h3>
        <pre><code id="sql-code"></code></pre>
    </div>
    <script>
        const chatId = "session_" + Math.random().toString(36).substring(2, 9);
        document.getElementById('room-display').textContent = chatId;
        const ws = new WebSocket("ws://" + window.location.host + "/ws/chat/" + chatId);
        const chatBox = document.getElementById('chat-box');

        function appendMessage(cls, label, text) {
            const p = document.createElement('p');
            p.className = cls;
            const b = document.createElement('b');
            b.textContent = label;
            p.appendChild(b);
            // FIX: Use a text node instead of innerHTML to prevent XSS
            // from model-generated content containing HTML or script tags.
            p.appendChild(document.createTextNode(' ' + text));
            chatBox.appendChild(p);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            appendMessage('agent-msg', 'Agent:', data.message);

            const s = data.system_status;
            document.getElementById('slot-widget').textContent = s.widget || "None";
            document.getElementById('slot-query').textContent = s.query ? "Filled" : "Empty";

            // Completeness meter
            const pct = (s.completeness || 0);
            const pctDisplay = pct.toFixed(2);
            document.getElementById('slot-completeness').textContent = pctDisplay;
            const bar = document.getElementById('completeness-bar');
            bar.style.width = (pct * 100).toFixed(0) + '%';
            const gate = document.getElementById('slot-completeness-gate');
            if (pct > 0.8) {
                bar.style.background = '#4caf50';
                gate.textContent = ' ✓ threshold met';
                gate.style.color = '#4caf50';
            } else {
                bar.style.background = pct > 0.5 ? '#ff9800' : '#888';
                gate.textContent = ' (needs > 0.80)';
                gate.style.color = '#888';
            }

            if (s.sql) {
                document.getElementById('sql-code').textContent = s.sql;
                document.getElementById('sql-output-area').style.display = 'block';
            }
        };

        ws.onerror = function() {
            appendMessage('agent-msg', 'System:', 'Connection error. Please refresh the page.');
        };

        ws.onclose = function() {
            appendMessage('agent-msg', 'System:', 'Session ended.');
            document.getElementById('send-btn').disabled = true;
        };

        function sendMessage() {
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text) return;
            if (ws.readyState !== WebSocket.OPEN) {
                appendMessage('agent-msg', 'System:', 'Connection lost. Please refresh.');
                return;
            }
            ws.send(text);
            appendMessage('user-msg', 'You:', text);
            input.value = '';
        }

        document.getElementById('send-btn').addEventListener('click', sendMessage);
        document.getElementById('user-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""


@app.get("/")
async def get_index():
    return HTMLResponse(html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
