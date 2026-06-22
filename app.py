import json
import os
import sqlite3
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessagesTypeAdapter, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from pydantic_ai.output import NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer

import config
from system_prompts import build_intake_prompt

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

# FIX 1/3: DB paths are explicit constants; admin db and data db are separate.
# get_db_connection defaults to ADMIN_DB_PATH so all admin operations (chat
# history, widget registry) never accidentally hit the data db.
DB_FILE_PATH  = "db_setup/data.db"
ADMIN_DB_PATH = "db_setup/dashboard_system.db"


# =====================================================================
# 1. DB HELPERS
# =====================================================================

@contextmanager
def get_db_connection(db_path: str = ADMIN_DB_PATH):   # FIX 3: default → admin db
    """Context manager that always commits or rolls back and closes."""
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
    """Upsert the full serialised message history for a chat session.

    FIX 7/9/10: The old log_chat_message(chat_id, sender, message) mixed two
    incompatible responsibilities — logging individual messages AND persisting
    the full history blob.  The schema uses an UPSERT on chat_id and stores the
    entire history as a single JSON column, so only one function is needed.
    All callers now pass the already-serialised JSON string.
    """
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
    """Return the full pydantic-ai message list for a session, or [] if new.

    FIX 13: ModelMessagesTypeAdapter and json are now imported at the top.
    FIX 3:  Explicitly uses ADMIN_DB_PATH.
    """
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
    """Insert or update a widget record in widget_query_mapping.

    FIX 6: update_widget was called but never defined.
    Note: dashboard_widget_mapping requires a dashboard_id + widget_id FK pair,
    so this writes only to widget_query_mapping which has widget_id as PK.
    Wire up dashboard assignment separately once you have a user/session → dashboard mapping.
    """
    with get_db_connection(ADMIN_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO widget_query_mapping
                (widget_id, query, status, user_agent_conversation, widget_type)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(widget_id) DO UPDATE SET
                query                 = excluded.query,
                status                = excluded.status,
                user_agent_conversation = excluded.user_agent_conversation,
                widget_type           = excluded.widget_type
            """,
            (widget_id, query, status, user_conversation, widget_type),
        )
    logger.info("Widget %s upserted (type=%s, status=%s).", widget_id, widget_type, status)


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


class IntakeAgentDeps:
    def __init__(self, schema: DataSourceSchema, table: List[TableDescription]):
        self.schema = schema
        self.table = table          # FIX 11: stored as .table (not .tables)


def generate_registry_from_metadata(
    db_path: str,
) -> Tuple[DataSourceSchema, List[TableDescription]]:
    """Load schema + table descriptions from the admin metadata tables.

    FIX 2: Early-return path now always returns the expected 2-tuple so callers
    never have to handle two different return shapes.
    """
    if not os.path.exists(db_path):
        logger.warning(
            "Metadata database '%s' not found. Defaulting to empty schema.", db_path
        )
        return DataSourceSchema(source_name="empty_db", tables=[]), []   # FIX 2

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


# FIX 1: was called with undefined `DB_NAME`; now uses ADMIN_DB_PATH
db_schema, table_desc_deps = generate_registry_from_metadata(ADMIN_DB_PATH)
dependencies        = SQLAgentDeps(schema=db_schema)
dependencies_intake = IntakeAgentDeps(schema=db_schema, table=table_desc_deps)


# =====================================================================
# 3. STRUCTURED OUTPUT SCHEMA
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
        description="True only when the user has explicitly agreed to proceed.",
    )
    reply: str = Field(description="Your next conversational message to the user.")


# =====================================================================
# 4. AGENT DEFINITIONS
# =====================================================================

# Corrected profile: self-hosted Ollama v0.5+ enforces JSON schema at the
# sampler level via llama.cpp grammars, but pydantic-ai's built-in profile
# for qwen2.5 doesn't set supports_json_schema_output.  We override it so
# NativeOutput routes through Ollama's response_format API instead of prompt
# injection, giving token-level schema enforcement on a 7B model.
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

# No static system_prompt — the full prompt (with schema embedded) is built
# dynamically on every turn by inject_metadata(dynamic=True) below, so the
# model always sees the real table/column list right next to the rules that
# reference it, regardless of whether message_history is being replayed.
intake_agent = Agent(
    model=local_qwen_model,
    output_type=NativeOutput(IntakeOutput),
    deps_type=IntakeAgentDeps,
)

sql_query_agent = Agent(
    model=local_qwen_model,
    deps_type=SQLAgentDeps,
    retries=3,
    system_prompt=(
        "You are a strict SQL Generator. "
        "Output ONLY a raw executable SQLite statement. Do not explain anything."
    ),
)


@sql_query_agent.system_prompt
def inject_database_schema(ctx: RunContext[SQLAgentDeps]) -> str:
    schema = ctx.deps.schema
    if not schema.tables:
        return "WARNING: No tables found. You cannot generate a valid SQL query."
    lines = [f"Write SQLite for: '{schema.source_name}'.", "Schema:"]
    for table in schema.tables:
        lines.append(f"Table: {table.table_name}")
        for col in table.columns:
            lines.append(f"  - {col.name} ({col.type})")
    return "\n".join(lines)


@intake_agent.system_prompt(dynamic=True)
def inject_metadata(ctx: RunContext[IntakeAgentDeps]) -> str:
    """Build and return the full unified system prompt on every turn.

    dynamic=True means pydantic-ai re-runs this function even when
    message_history is provided, so the schema is always fresh and
    always embedded inline next to the rules that reference it.
    The schema is no longer a trailing appendix — it lives inside the
    data-questions rule, eliminating the attention gap that caused the
    model to ignore it on small-model inference.
    """
    return build_intake_prompt(
        tables=ctx.deps.schema.tables,
        table_descriptions=ctx.deps.table,
    )


# =====================================================================
# 5. MESSAGE HISTORY HELPERS
# =====================================================================

def serialise_history(messages: list) -> str:
    """Serialise a pydantic-ai message list to a JSON string for DB storage."""
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

    # Load persisted history or start fresh
    message_history = load_chat_history(chat_id)
    is_new_chat = len(message_history) == 0

    if is_new_chat:
        greeting = (
            "Hello! Let's configure a dashboard widget. "
            "What kind of layout are we building? (e.g., KPI, BARCHART)"
        )
        await websocket.send_json({
            "sender": "agent",
            "message": greeting,
            "system_status": {"widget": None, "query": None, "confirmed": False,
                              "completeness": 0.0, "sql": None},
        })
        # Persist the greeting as the first history entry
        # FIX 10: use .model_dump() not .dump()
        seed = ModelRequest(parts=[UserPromptPart(content="")])   # placeholder so history is non-empty
        # We store the greeting as a ModelResponse so the agent sees it as prior context
        greeting_msg = ModelResponse(parts=[TextPart(content=greeting)])
        message_history = [greeting_msg]
        log_chat_history(chat_id, serialise_history(message_history))
    else:
        # Replay prior conversation to the frontend on reconnect.
        # Walk the pydantic-ai message list in order and re-emit each
        # user and assistant turn as a chat event so the UI can rebuild
        # the conversation thread without any extra state of its own.
        for msg in message_history:
            if msg.kind == "request":
                # Extract user-prompt parts only (skips injected system prompts)
                for part in msg.parts:
                    if part.part_kind == "user-prompt" and part.content:
                        await websocket.send_json({
                            "sender": "user",
                            "message": part.content,
                            "system_status": None,
                        })
            elif msg.kind == "response":
                # Extract text parts only (skips tool-call / tool-return parts)
                for part in msg.parts:
                    if part.part_kind == "text" and part.content:
                        await websocket.send_json({
                            "sender": "agent",
                            "message": part.content,
                            "system_status": None,
                        })
        logger.info(
            "[%s] Replayed %d history messages to frontend on reconnect.",
            chat_id, len(message_history),
        )

    try:
        while True:
            user_message = await websocket.receive_text()

            # ── Intake agent ───────────────────────────────────────────
            try:
                logger.info("[%s] Calling Intake Agent...", chat_id)

                result = await intake_agent.run(
                    user_message,
                    message_history=message_history,
                    deps=dependencies_intake,
                    model_settings={"temperature": 0.1},
                )

                # FIX 4: replace history list, don't append to it
                message_history = result.all_messages()

                intake: IntakeOutput = result.output

                # ── Server-side confirmation safety net ────────────────
                # If the model returned is_confirmed=False despite a clear
                # affirmative and both fields populated, override it here.
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
                        "[%s] Model missed confirmation — overriding is_confirmed=True "
                        "(user said: %r)", chat_id, user_message,
                    )
                    intake = intake.model_copy(update={"is_confirmed": True, "completeness": 0.9})

                logger.info(
                    "[%s] Intake → completeness=%s | widget=%s | query=%s | confirmed=%s",
                    chat_id, intake.completeness, intake.widget_type,
                    intake.query_description, intake.is_confirmed,
                )

            except Exception as e:
                logger.error("[%s] Intake agent failure — %s: %s", chat_id, type(e).__name__, e)
                await websocket.send_json({
                    "sender": "agent",
                    "message": "Processing issue encountered. Let's try that again.",
                    "system_status": {
                        "widget": None, "query": None,
                        "confirmed": False, "completeness": 0.0, "sql": None,
                    },
                })
                continue

            # ── SQL generation ─────────────────────────────────────────
            generated_sql = None

            if intake.completeness > 0.8 and intake.is_confirmed and intake.query_description:
                if not db_schema.tables:
                    generated_sql = "-- Error: No database tables found. Cannot generate SQL."
                    logger.warning("[%s] SQL generation skipped: empty schema.", chat_id)
                else:
                    logger.info(
                        "[%s] completeness=%s > 0.8 — generating SQL...",
                        chat_id, intake.completeness,
                    )
                    try:
                        sql_result = await sql_query_agent.run(
                            f"Generate an accurate SQLite query for: {intake.query_description}",
                            deps=dependencies,
                            model_settings={"temperature": 0.1},
                        )
                        generated_sql = sql_result.output.strip()
                        logger.info("[%s] SQL generated successfully.", chat_id)
                    except Exception as sql_err:
                        logger.error("[%s] SQL generation failed: %s", chat_id, sql_err)
                        generated_sql = f"-- Execution error: {sql_err}"

                    # FIX 5/6: uuid4 is a callable, not a class; update_widget now defined
                    if generated_sql and not generated_sql.startswith("--"):
                        widget_id = str(uuid4())
                        update_widget(
                            widget_id=widget_id,
                            user_conversation=intake.query_description,
                            query=generated_sql,
                            status="NEW",
                            widget_type=intake.widget_type,
                        )

                # FIX 14: reset history after a completed cycle so the next widget
                # starts a fresh conversation instead of carrying stale context
                message_history = []

            # ── Persist history & respond ──────────────────────────────
            # FIX 7: serialise_history handles the list→JSON conversion correctly
            log_chat_history(chat_id, serialise_history(message_history))

            await websocket.send_json({
                "sender": "agent",
                "message": intake.reply,
                "system_status": {
                    "widget": intake.widget_type,
                    "query": intake.query_description,
                    "confirmed": intake.is_confirmed,
                    "completeness": intake.completeness,
                    "sql": generated_sql,
                },
            })

    except WebSocketDisconnect:
        logger.warning("Session closed for: [%s]", chat_id)


# =====================================================================
# 7. FRONTEND
# =====================================================================

@app.get("/")
async def get_index():
    """Serve the test frontend.
    Looks for frontend.html next to this file; falls back to a minimal
    inline page so the server stays usable even without the asset.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(here, "frontend.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(
        "<h2 style='font-family:sans-serif;padding:2rem'>"
        "frontend.html not found — place it next to app.py</h2>",
        status_code=200,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
