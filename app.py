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
ADMIN_DB_PATH = "db_setup/dashboard_system.db"

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

def log_chat_message(chat_id: str, sender: str, message: str):
    """Inserts a structured interaction frame into the database table."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_message_history (chat_id, sender, message) VALUES (?, ?, ?);",
            (chat_id, sender, message)
        )
    logger.info(f"[{chat_id}] Transaction logged to DB -> {sender.upper()}: {message[:50]}...")

# =====================================================================
# 2. SCHEMA DISCOVERY FOR THE SQL AGENT
# =====================================================================

class ColumnSchema(BaseModel):
    name: str
    type: str
    description: str


class TableSchema(BaseModel):
    table_name: str
    columns: List[ColumnSchema]


class DataSourceSchema(BaseModel):
    source_name: str
    tables: List[TableSchema]


class SQLAgentDeps:
    def __init__(self, schema: DataSourceSchema):
        self.schema = schema

