"""routers/chat.py

Endpoints for inspecting and clearing persisted chat sessions.

Table used:
  chat_message_history (chat_id, message_history TEXT, timestamp)
"""
import json
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from database import get_db

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatSession(BaseModel):
    chat_id:   str
    timestamp: str
    message_count: int   # number of pydantic-ai messages in the blob


class ChatHistory(BaseModel):
    chat_id:  str
    timestamp: str
    messages: List[Any]  # raw pydantic-ai message dicts


@router.get("", response_model=List[ChatSession])
def list_sessions():
    """List all chat sessions ordered newest-first."""
    with get_db() as db:
        rows = db.execute(
            "SELECT chat_id, message_history, timestamp FROM chat_message_history ORDER BY timestamp DESC"
        ).fetchall()

    result = []
    for row in rows:
        try:
            msgs = json.loads(row["message_history"])
            count = len(msgs) if isinstance(msgs, list) else 0
        except (ValueError, TypeError):
            count = 0
        result.append(ChatSession(
            chat_id=row["chat_id"],
            timestamp=row["timestamp"],
            message_count=count,
        ))
    return result


@router.get("/{chat_id}", response_model=ChatHistory)
def get_session(chat_id: str):
    """Return the full message history for a chat session."""
    with get_db() as db:
        row = db.execute(
            "SELECT chat_id, message_history, timestamp FROM chat_message_history WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{chat_id}' not found.")

    try:
        messages = json.loads(row["message_history"])
    except (ValueError, TypeError):
        messages = []

    return ChatHistory(
        chat_id=row["chat_id"],
        timestamp=row["timestamp"],
        messages=messages,
    )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(chat_id: str):
    """Delete a chat session's history so the next connection starts fresh."""
    with get_db() as db:
        result = db.execute(
            "DELETE FROM chat_message_history WHERE chat_id = ?", (chat_id,)
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Session '{chat_id}' not found.")
