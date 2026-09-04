import json
import os
import sqlite3
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple
from uuid import uuid4
import config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.output import NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer

from system_prompts import build_intake_prompt
from routers import dashboards as dashboards_router
from routers import metadata   as metadata_router
from routers import chat       as chat_router

# =====================================================================
# 0. LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("DashboardAgent")

app = FastAPI(title="Dynamic SQL Dashboard Agent API")

app.include_router(dashboards_router.router)
app.include_router(metadata_router.router)
app.include_router(chat_router.router)

DB_FILE_PATH  = "db_setup/data.db"
ADMIN_DB_PATH = "db_setup/dashboard_system.db"


# =====================================================================
# 1. DB HELPERS
# =====================================================================

@contextmanager
def get_db_connection(db_path: str = ADMIN_DB_PATH):
    connection = sqlite3.connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def log_chat_history(chat_id: str, history_json: str) -> None:
    with get_db_connection(ADMIN_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chat_message_history (chat_id, message_history)
            VALUES (?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                message_history = excluded.message_history,
                timestamp       = CURRENT_TIMESTAMP
            """,
            (chat_id, history_json),
        )
    logger.info("[%s] Chat history persisted (%d bytes).", chat_id, len(history_json))


def load_chat_history(chat_id: str):
    with get_db_connection(ADMIN_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_history FROM chat_message_history WHERE chat_id = ?",
            (chat_id,),
        )
        row = cursor.fetchone()
    if not row:
        return []
    return ModelMessagesTypeAdapter.validate_json(row[0])


def update_widget(
    widget_id: str,
    user_conversation: str,
    query: str,
    status: str,
    widget_type: str,
) -> None:
    with get_db_connection(ADMIN_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO widget_query_mapping
                (widget_id, query, status, user_agent_conversation, widget_type)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(widget_id) DO UPDATE SET
                query                   = excluded.query,
                status                  = excluded.status,
                user_agent_conversation = excluded.user_agent_conversation,
                widget_type             = excluded.widget_type
            """,
            (widget_id, query, status, user_conversation, widget_type),
        )
    logger.info("Widget %s upserted (type=%s).", widget_id, widget_type)


def execute_query(query: str, db_path: str = DB_FILE_PATH) -> list:
    """Run a SELECT against the data db and return rows as list of dicts.

    FIX 5: The db_path parameter was previously ignored — the body always
    used DB_FILE_PATH directly. Now it honours the parameter correctly.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


# =====================================================================
# 2. SCHEMA DISCOVERY
# =====================================================================

class ColumnSchema(BaseModel):
    name: str
    type: str
    description: Optional[str] = None


class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnSchema]


class DataSourceSchema(BaseModel):
    source_name: str
    tables: List[TableSchema]


class TableDescription(BaseModel):
    table_name: str
    description: Optional[str] = None


class SQLAgentDeps:
    def __init__(self, schema: DataSourceSchema):
        self.schema = schema


def generate_registry_from_metadata(
    db_path: str,
) -> Tuple[DataSourceSchema, List[TableDescription]]:
    if not os.path.exists(db_path):
        logger.warning("Metadata database '%s' not found.", db_path)
        return DataSourceSchema(source_name="empty_db", tables=[]), []

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT table_name, column_name, column_type, column_description
            FROM   metadata_data_table
            ORDER  BY table_name, id
            """
        )
        tables_dict: dict = defaultdict(list)
        for table_name, column_name, column_type, description in cursor.fetchall():
            tables_dict[table_name].append(
                ColumnSchema(name=column_name, type=column_type, description=description)
            )
        table_schemas = [
            TableSchema(table_name=t, columns=cols)
            for t, cols in tables_dict.items()
        ]
        cursor.execute(
            """
            SELECT table_name, table_description
            FROM   metadata_data_table_description
            ORDER  BY table_name, id
            """
        )
        table_desc = [
            TableDescription(table_name=t, description=d)
            for t, d in cursor.fetchall()
        ]

    schema_res = DataSourceSchema(
        source_name=os.path.splitext(os.path.basename(db_path))[0],
        tables=table_schemas,
    )
    logger.info("Metadata schema loaded. Tables: %s", [t.table_name for t in schema_res.tables])
    return schema_res, table_desc


db_schema, table_desc_deps = generate_registry_from_metadata(ADMIN_DB_PATH)

INTAKE_SYSTEM_PROMPT: str = build_intake_prompt(
    tables=db_schema.tables,
    table_descriptions=table_desc_deps,
)
logger.info("Intake system prompt built (%d chars).", len(INTAKE_SYSTEM_PROMPT))


def build_sql_system_prompt(schema: DataSourceSchema) -> str:
    """Build the SQL agent system prompt with schema embedded at startup.

    FIX 8: The old system prompt said "Output ONLY a raw SQLite statement"
    while the output_type was NativeOutput(SQLOutput) — direct contradiction.
    The prompt now correctly describes the JSON output format the model must
    produce, matching the SQLOutput schema exactly.
    """
    lines = [
        "You are a SQL generator for SQLite databases.",
        "You must return a JSON object with exactly these fields:",
        "  sql_query            : the raw executable SQLite SELECT statement",
        "  confidence           : float 0.0–1.0 (1.0 = certain, 0.0 = unsure)",
        "  possible_data_errors : string describing potential issues, or null if none",
        "",
        "Rules:",
        "  - Use ONLY the tables and columns listed in the schema below.",
        "  - Do NOT invent columns or tables.",
        "  - If the request cannot be satisfied with this schema, set sql_query to",
        '    "INSUFFICIENT DATAPOINTS" and confidence to 0.0.',
        "  - Never include LIMIT in sql_query — it will be applied separately.",
        "",
    ]
    if not schema.tables:
        lines.append("WARNING: No tables available.")
    else:
        lines.append(f"Database: {schema.source_name}")
        lines.append("Schema:")
        for table in schema.tables:
            lines.append(f"  Table: {table.table_name}")
            for col in table.columns:
                col_line = f"    - {col.name} ({col.type})"
                if col.description:
                    col_line += f"  — {col.description}"
                lines.append(col_line)
    return "\n".join(lines)


SQL_SYSTEM_PROMPT: str = build_sql_system_prompt(db_schema)
logger.info("SQL system prompt built (%d chars).", len(SQL_SYSTEM_PROMPT))


# =====================================================================
# 3. STRUCTURED OUTPUT SCHEMAS
# =====================================================================

ALLOWED_WIDGETS = {"KPI", "BARCHART", "PIECHART", "LINECHART", "STACKEDBAR"}


class IntakeState(BaseModel):
    """Server-side conversation state carried across turns within a session."""
    widget_type: Optional[str]       = None
    query_description: Optional[str] = None
    is_confirmed: bool               = False

    def reset(self) -> None:
        """Clear state after a completed widget cycle."""
        self.widget_type       = None
        self.query_description = None
        self.is_confirmed      = False


class IntakeOutput(BaseModel):
    completeness: Literal[0.0, 0.3, 0.5, 0.7, 0.9, 1.0] = Field(
        description=(
            "0.0=greeting/off-topic; 0.3=vague interest; "
            "0.5=one of widget/query known; 0.7=both known awaiting confirm; "
            "0.9=user confirmed; 1.0=fully done."
        ),
    )
    widget_type: Optional[str] = Field(
        default=None,
        description=f"One of {sorted(ALLOWED_WIDGETS)}, or null.",
    )
    query_description: Optional[str] = Field(
        default=None,
        description="The user's data query in plain English, or null.",
    )
    is_confirmed: bool = Field(
        default=False,
        description="True only when the user has explicitly confirmed.",
    )
    reply: str = Field(description="Your next message to the user.")


class SQLOutput(BaseModel):
    # FIX 6/7: Field() does not accept a positional string as description.
    # Positional strings become the DEFAULT VALUE, not the description.
    # All three fields now use the correct keyword argument.
    sql_query: str = Field(
        description="The raw executable SQLite SELECT statement, or 'INSUFFICIENT DATAPOINTS'."
    )
    confidence: float = Field(
        description="Confidence score 0.0–1.0. 1.0 = certain the query is correct."
    )
    possible_data_errors: Optional[str] = Field(
        default=None,
        description="Potential issues with the query (missing columns, type mismatches, etc.), or null."
    )


# =====================================================================
# 4. AGENT DEFINITIONS
# =====================================================================

_qwen25_local_profile = ModelProfile(
    json_schema_transformer=InlineDefsJsonSchemaTransformer,
    ignore_streamed_leading_whitespace=True,
    supports_json_schema_output=True,
    supports_json_object_output=True,
)

local_qwen_model = OllamaModel(
    model_name="qwen2.5:7b-instruct",
    profile=_qwen25_local_profile,
)

intake_agent = Agent(
    model=local_qwen_model,
    output_type=NativeOutput(IntakeOutput),
    system_prompt=INTAKE_SYSTEM_PROMPT,
)

sql_query_agent = Agent(
    model=local_qwen_model,
    output_type=NativeOutput(SQLOutput),
    retries=3,
    system_prompt=SQL_SYSTEM_PROMPT,
)


# =====================================================================
# 5. MESSAGE HISTORY HELPERS
# =====================================================================

def serialise_history(messages: list) -> str:
    return ModelMessagesTypeAdapter.dump_json(messages).decode()


# =====================================================================
# 6. WEBSOCKET ENDPOINT
# =====================================================================

_AFFIRMATIVES = frozenset({
    "yes", "y", "yep", "yeah", "yup", "ok", "okay", "sure",
    "confirm", "confirmed", "correct", "go ahead", "proceed",
    "looks good", "that's right", "thats right", "do it",
    "go for it", "absolutely", "affirmative", "agreed",
})


@app.websocket("/ws/chat/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    await websocket.accept()
    logger.info("WebSocket session open: [%s]", chat_id)

    message_history = load_chat_history(chat_id)
    is_new_chat = len(message_history) == 0

    # FIX 3: IntakeState is initialised once per connection and updated
    # BEFORE each prompt is built so the model always sees current values.
    # On reconnect we can't recover state from the DB (it's not persisted
    # separately), so state resets — acceptable because the stateless prompt
    # approach already carries widget/query forward in the model's reply context.
    state = IntakeState()

    if is_new_chat:
        greeting = (
            "Hello! Let's configure a dashboard widget. "
            "What kind of layout are we building? (e.g., KPI, BARCHART)"
        )
        await websocket.send_json({
            "sender": "agent",
            "message": greeting,
            "system_status": {
                "widget": None, "query": None,
                "confirmed": False, "completeness": 0.0,
                "sql": None, "result": [],
            },
        })
        greeting_msg = ModelResponse(parts=[TextPart(content=greeting)])
        message_history = [greeting_msg]
        log_chat_history(chat_id, serialise_history(message_history))
    else:
        # FIX 9: Replay prior conversation to the frontend on reconnect.
        for msg in message_history:
            if msg.kind == "request":
                for part in msg.parts:
                    if part.part_kind == "user-prompt" and part.content:
                        await websocket.send_json({
                            "sender": "user",
                            "message": part.content,
                            "system_status": None,
                        })
            elif msg.kind == "response":
                for part in msg.parts:
                    if part.part_kind == "text" and part.content:
                        await websocket.send_json({
                            "sender": "agent",
                            "message": part.content,
                            "system_status": None,
                        })
        logger.info("[%s] Replayed %d messages.", chat_id, len(message_history))

    try:
        while True:
            user_message = await websocket.receive_text()

            # ── Intake agent ───────────────────────────────────────────
            try:
                logger.info("[%s] Calling Intake Agent...", chat_id)

                # FIX 3: State is injected into the prompt BEFORE running the
                # agent so the model sees the values set on the PREVIOUS turn,
                # not stale values from two turns ago.
                prompt = (
                    f"Current State:\n"
                    f"  widget_type          = {state.widget_type}\n"
                    f"  query_description    = {state.query_description}\n"
                    f"  awaiting_confirmation = {state.is_confirmed is False and (state.widget_type is not None and state.query_description is not None)}\n"
                    f"\nUser Message:\n{user_message}"
                )
                logger.info("[%s] Prompt:\n%s", chat_id, prompt)

                result = await intake_agent.run(
                    prompt,
                    model_settings={"temperature": 0.1},
                )

                intake: IntakeOutput = result.output

                # Server-side confirmation safety net
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
                        "[%s] Overriding is_confirmed=True (user said: %r)",
                        chat_id, user_message,
                    )
                    intake = intake.model_copy(
                        update={"is_confirmed": True, "completeness": 0.9}
                    )

                # FIX 3: Update state immediately after getting intake output,
                # before any other logic, so the next turn's prompt is correct.
                state.widget_type       = intake.widget_type
                state.query_description = intake.query_description
                state.is_confirmed      = intake.is_confirmed

                logger.info(
                    "[%s] Intake → completeness=%s | widget=%s | confirmed=%s",
                    chat_id, intake.completeness, intake.widget_type, intake.is_confirmed,
                )

            except Exception as e:
                logger.error("[%s] Intake failure — %s: %s", chat_id, type(e).__name__, e)
                await websocket.send_json({
                    "sender": "agent",
                    "message": "Processing issue encountered. Let's try that again.",
                    "system_status": {
                        "widget": None, "query": None,
                        "confirmed": False, "completeness": 0.0,
                        "sql": None, "result": [],
                    },
                })
                continue

            # ── SQL generation ─────────────────────────────────────────
            generated_sql   = None
            sql_confidence  = None
            sql_data_errors = None
            query_result: list = []

            if intake.completeness > 0.8 and intake.is_confirmed and intake.query_description:
                if not db_schema.tables:
                    generated_sql = "-- Error: No database tables found."
                    logger.warning("[%s] SQL skipped: empty schema.", chat_id)
                else:
                    try:
                        logger.info("[%s] Generating SQL for: %s",
                                    chat_id, intake.query_description)
                        sql_result = await sql_query_agent.run(
                            f"Generate an accurate SQLite query for: {intake.query_description}",
                            model_settings={"temperature": 0.1},
                        )
                        sql_output: SQLOutput = sql_result.output
                        generated_sql   = sql_output.sql_query.strip()
                        sql_confidence  = sql_output.confidence
                        sql_data_errors = sql_output.possible_data_errors

                        logger.info(
                            "[%s] SQL generated. confidence=%.2f errors=%s",
                            chat_id, sql_confidence or 0, sql_data_errors,
                        )
                    except Exception as sql_err:
                        logger.error("[%s] SQL failed: %s", chat_id, sql_err)
                        generated_sql = f"-- Error: {sql_err}"

                    # FIX 2: LIMIT is applied only to the execution call, never
                    # stored in the DB. The clean SQL is what gets persisted.
                    if generated_sql and not generated_sql.startswith(("--", "INSUFFICIENT")):
                        update_widget(
                            widget_id=str(uuid4()),
                            user_conversation=intake.query_description,
                            query=generated_sql,          # clean, no LIMIT
                            status="NEW",
                            widget_type=intake.widget_type,
                        )
                        # FIX 1: execute_query only called when we have valid SQL.
                        # LIMIT appended here only for the data fetch, not stored.
                        try:
                            query_result = execute_query(
                                generated_sql + " LIMIT 100",
                                db_path=DB_FILE_PATH,
                            )
                            logger.info("[%s] Query returned %d rows.", chat_id, len(query_result))
                        except Exception as qerr:
                            logger.error("[%s] Query execution failed: %s", chat_id, qerr)
                            query_result = []

                # FIX 10: Reset state after a completed widget cycle so the next
                # conversation starts clean instead of seeing is_confirmed=True.
                state.reset()
                message_history = []

            log_chat_history(chat_id, serialise_history(message_history))

            await websocket.send_json({
                "sender": "agent",
                "message": intake.reply,
                "system_status": {
                    "widget":      intake.widget_type,
                    "query":       intake.query_description,
                    "confirmed":   intake.is_confirmed,
                    "completeness": intake.completeness,
                    "sql":         generated_sql,
                    "result":      query_result,
                    # Extra diagnostic fields — useful in the Event Log
                    "sql_confidence":  sql_confidence,
                    "sql_data_errors": sql_data_errors,
                },
            })

    except WebSocketDisconnect:
        logger.warning("Session closed: [%s]", chat_id)


# =====================================================================
# 7. FRONTEND
# =====================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


def _serve_html(filename: str):
    path = os.path.join(_HERE, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return HTMLResponse(f"<h2 style='font-family:sans-serif;padding:2rem'>{filename} not found.</h2>")


@app.get("/")
async def get_landing():
    return _serve_html("index.html")


@app.get("/builder")
async def get_builder():
    return _serve_html("frontend.html")


@app.get("/dashboards.html")
async def get_dashboards_page():
    return _serve_html("dashboards.html")


@app.get("/data_steward.html")
async def get_data_steward_page():
    return _serve_html("data_steward.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
