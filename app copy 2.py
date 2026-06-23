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
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from pydantic_ai.output import NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer

from system_prompts import build_intake_prompt
import config

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


def execute_query(query, db_path=DB_FILE_PATH):
    with get_db_connection(DB_FILE_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(query)

        columns = [c[0] for c in cursor.description]

        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
def generate_registry_from_metadata(
    db_path: str,
) -> Tuple[DataSourceSchema, List[TableDescription]]:
    if not os.path.exists(db_path):
        logger.warning("Metadata database '%s' not found. Defaulting to empty schema.", db_path)
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
sql_deps = SQLAgentDeps(schema=db_schema)

# Build the intake system prompt once at startup with the schema embedded.
# Rebuilding it here means the model receives a single self-contained system
# prompt — no decorator injection, no deps plumbing, no attention-gap between
# the rules and the data they reference.
INTAKE_SYSTEM_PROMPT: str = build_intake_prompt(
    tables=db_schema.tables,
    table_descriptions=table_desc_deps,
)
logger.info("Intake system prompt built (%d chars).", len(INTAKE_SYSTEM_PROMPT))
logger.info(f"Full prompt {INTAKE_SYSTEM_PROMPT}")

# =====================================================================
# 3. STRUCTURED OUTPUT SCHEMA
# =====================================================================

ALLOWED_WIDGETS = {"KPI", "BARCHART", "PIECHART", "LINECHART", "STACKEDBAR"}


class IntakeState(BaseModel):
    widget_type:Optional[str] = Field(
        default=None)
    query_description:Optional[str] = Field(
        default=None)
    is_confirmed:bool = Field(default = False)




class IntakeOutput(BaseModel):
    completeness: Literal[0.0, 0.3, 0.5, 0.7, 0.9, 1.0] = Field(
        description=(
            "How complete the user's data-visualisation request is. "
            "0.0 = greeting/off-topic; 0.3 = vague interest; "
            "0.5 = one of widget/query known; 0.7 = both known, awaiting confirm; "
            "0.9 = user confirmed; 1.0 = fully done."
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
    sql_query: str = Field("The sql query based on the user`s query description")
    confidence: float = Field("The confidence based on query description and the query generated. 1.0 - High confidence, 0.0 - low confidence")
    possible_data_errors: Optional[str] = Field("The errors that the query can lead to. For eg: missing columns, wrong data type, logical or semantic flaws etc.")


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

# Intake agent: no deps, no decorator — just a plain system_prompt string
# that already contains the schema. Simple, transparent, debuggable.
intake_agent = Agent(
    model=local_qwen_model,
    output_type=NativeOutput(IntakeOutput),
    system_prompt=INTAKE_SYSTEM_PROMPT,
)

# SQL agent: deps still used here because the schema is referenced at
# query-time and may need to be re-injected cleanly per request.
# @sql_query_agent.system_prompt
def inject_database_schema(schema) -> str:
    if not schema.tables:
        return "WARNING: No tables found. You cannot generate a valid SQL query."
    lines = [f"Write SQLite for: '{schema.source_name}'.", "Schema:"]
    for table in schema.tables:
        lines.append(f"Table: {table.table_name}")
        for col in table.columns:
            lines.append(f"  - {col.name} ({col.type})")
    return "\n".join(lines)


sql_query_agent = Agent(
    model=local_qwen_model,
    retries=3,
    output_type= NativeOutput(SQLOutput),
    system_prompt=(
        "You are a strict SQL Generator. "
        "Output ONLY a raw executable SQLite statement. Do not explain anything."
        "You must follow the following schema only and donnot make up columns."
        "If the current schema doesnt suffice respond with INSUFFICIENT DATAPOINTS"
        f"{inject_database_schema(db_schema)}"
    ),
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
                "confirmed": False, "completeness": 0.0, "sql": None,
            },
        })
        greeting_msg = ModelResponse(parts=[TextPart(content=greeting)])
        message_history = [greeting_msg]
        log_chat_history(chat_id, serialise_history(message_history))
    
    try:
        while True:
            user_message = await websocket.receive_text()

            # ── Intake agent ───────────────────────────────────────────
            try:
                logger.info("[%s] Calling Intake Agent ...",
                            chat_id)

                prompt = f"""
                Current State:
                widget_type={state.widget_type}
                query_description={state.query_description}
                awaiting_confirmation={not state.is_confirmed}

                User Message:
                {user_message}
                """
                logger.info(prompt)

                    
                result = await intake_agent.run(
                    prompt,
                    # message_history=message_history,
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
                        "confirmed": False, "completeness": 0.0, "sql": None,
                    },
                })
                continue

            # ── SQL generation ─────────────────────────────────────────
            generated_sql = None

            if intake.completeness > 0.8 and intake.is_confirmed and intake.query_description:
                if not db_schema.tables:
                    generated_sql = "-- Error: No database tables found."
                    logger.warning("[%s] SQL skipped: empty schema.", chat_id)
                else:
                    try:
                        logger.info(f"Generate an accurate SQLite query for: {intake.query_description}")
                        sql_result = await sql_query_agent.run(
                            f"Generate an accurate SQLite query for: {intake.query_description}",
                            model_settings={"temperature": 0.1},
                        )
                        sql_output: SQLOutput = sql_result.output
                        generated_sql = sql_output.sql_query.strip()
                        print(generated_sql)
                        logger.info(sql_output.confidence)
                        
                        logger.info("[%s] SQL generated.", chat_id)
                    except Exception as sql_err:
                        logger.error("[%s] SQL failed: %s", chat_id, sql_err)
                        generated_sql = f"-- Error: {sql_err}"

                    if generated_sql and not generated_sql.startswith("--"):
                        update_widget(
                            widget_id=str(uuid4()),
                            user_conversation=intake.query_description,
                            query=generated_sql,
                            status="NEW",
                            widget_type=intake.widget_type,
                        )
                        generated_sql += " LIMIT 100"

                message_history = []

            log_chat_history(chat_id, serialise_history(message_history))
            result = []
            try:
                result = execute_query(query=generated_sql)
            except Exception as err:
                logger.error(str(err))


            await websocket.send_json({
                "sender": "agent",
                "message": intake.reply,
                "system_status": {
                    "widget": intake.widget_type,
                    "query": intake.query_description,
                    "confirmed": intake.is_confirmed,
                    "completeness": intake.completeness,
                    "sql": generated_sql,
                    "result":result
                },
            })
            state.widget_type = intake.widget_type
            state.query_description = intake.query_description
            state.is_confirmed = intake.is_confirmed

    except WebSocketDisconnect:
        logger.warning("Session closed: [%s]", chat_id)


# =====================================================================
# 7. FRONTEND
# =====================================================================

@app.get("/")
async def get_index():
    here = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(here, "frontend.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(
        "<h2 style='font-family:sans-serif;padding:2rem'>"
        "frontend.html not found — place it next to app.py</h2>",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
