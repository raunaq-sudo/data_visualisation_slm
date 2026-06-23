"""routers/metadata.py

Read-only endpoints exposing the metadata tables so the frontend
can build a schema browser or populate dropdowns without needing
direct DB access.

Tables used:
  metadata_data_table             (table_name, column_name, column_type, column_description)
  metadata_data_table_description (table_name, table_description)
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import get_db

router = APIRouter(prefix="/metadata", tags=["Metadata"])


# ── Response models ───────────────────────────────────────────────────

class ColumnMeta(BaseModel):
    column_name:        str
    column_type:        str
    column_description: Optional[str]


class TableMeta(BaseModel):
    table_name:        str
    table_description: Optional[str]
    columns:           List[ColumnMeta]


class TableSummary(BaseModel):
    table_name:        str
    table_description: Optional[str]
    column_count:      int


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("", response_model=List[TableSummary])
def list_tables():
    """Return all tables with their descriptions and column counts.

    Useful for a schema overview panel or table-picker dropdown.
    """
    with get_db() as db:
        descs = {
            row["table_name"]: row["table_description"]
            for row in db.execute(
                "SELECT table_name, table_description FROM metadata_data_table_description"
            ).fetchall()
        }
        counts = {
            row["table_name"]: row["cnt"]
            for row in db.execute(
                "SELECT table_name, COUNT(*) AS cnt FROM metadata_data_table GROUP BY table_name"
            ).fetchall()
        }

    return [
        TableSummary(
            table_name=t,
            table_description=descs.get(t),
            column_count=counts.get(t, 0),
        )
        for t in sorted(counts.keys())
    ]


@router.get("/tables", response_model=List[TableMeta])
def get_all_tables():
    """Return every table with its full column list."""
    with get_db() as db:
        descs = {
            row["table_name"]: row["table_description"]
            for row in db.execute(
                "SELECT table_name, table_description FROM metadata_data_table_description"
            ).fetchall()
        }
        col_rows = db.execute(
            """
            SELECT table_name, column_name, column_type, column_description
            FROM   metadata_data_table
            ORDER  BY table_name, id
            """
        ).fetchall()

    tables: dict = {}
    for row in col_rows:
        tn = row["table_name"]
        if tn not in tables:
            tables[tn] = TableMeta(
                table_name=tn,
                table_description=descs.get(tn),
                columns=[],
            )
        tables[tn].columns.append(ColumnMeta(
            column_name=row["column_name"],
            column_type=row["column_type"],
            column_description=row["column_description"],
        ))

    return list(tables.values())


@router.get("/tables/{table_name}", response_model=TableMeta)
def get_table(table_name: str):
    """Return a single table's description and full column list."""
    with get_db() as db:
        desc_row = db.execute(
            "SELECT table_description FROM metadata_data_table_description WHERE table_name = ?",
            (table_name,),
        ).fetchone()

        col_rows = db.execute(
            """
            SELECT column_name, column_type, column_description
            FROM   metadata_data_table
            WHERE  table_name = ?
            ORDER  BY id
            """,
            (table_name,),
        ).fetchall()

    if not col_rows:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in metadata.")

    return TableMeta(
        table_name=table_name,
        table_description=desc_row["table_description"] if desc_row else None,
        columns=[
            ColumnMeta(
                column_name=r["column_name"],
                column_type=r["column_type"],
                column_description=r["column_description"],
            )
            for r in col_rows
        ],
    )
