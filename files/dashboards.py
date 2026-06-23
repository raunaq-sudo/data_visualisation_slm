"""routers/dashboards.py

Dashboard + widget CRUD, canvas bulk-save.

Tables used:
  dashboard        (id, name, description, created_at, updated_at)
  dashboard_widget (id, dashboard_id, widget_uid, widget_type, title,
                    x, y, width, height, query, sql_query,
                    result_json, config_json, z_index, created_at, updated_at)
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from database import get_db

router = APIRouter(prefix="/dashboards", tags=["Dashboards"])


# ── Request / Response models ─────────────────────────────────────────

class DashboardCreate(BaseModel):
    name: str
    description: Optional[str] = None


class DashboardPatch(BaseModel):
    name: Optional[str]        = None
    description: Optional[str] = None


class DashboardRow(BaseModel):
    id:          int
    name:        str
    description: Optional[str]
    created_at:  str
    updated_at:  str


class WidgetCreate(BaseModel):
    widget_uid:  str  = Field(description="Client-generated UUID, stable across saves.")
    widget_type: str  = Field(description="KPI | BARCHART | PIECHART | LINECHART | STACKEDBAR")
    title:       Optional[str] = None
    x:           int  = Field(default=40,  description="Left offset in px on canvas.")
    y:           int  = Field(default=40,  description="Top offset in px on canvas.")
    width:       int  = Field(default=380, description="Width in px.")
    height:      int  = Field(default=240, description="Height in px.")
    z_index:     int  = Field(default=1)
    query:       Optional[str] = Field(default=None, description="User's natural-language query.")
    sql_query:   Optional[str] = Field(default=None, description="Generated SQL (no LIMIT).")
    result_json: Optional[Any] = Field(default=None, description="Query result rows as JSON.")
    config_json: Optional[Any] = Field(default=None, description="Extra chart config.")


class WidgetPatch(BaseModel):
    """All fields optional — supports partial updates from drag/resize events."""
    widget_type: Optional[str] = None
    title:       Optional[str] = None
    x:           Optional[int] = None
    y:           Optional[int] = None
    width:       Optional[int] = None
    height:      Optional[int] = None
    z_index:     Optional[int] = None
    query:       Optional[str] = None
    sql_query:   Optional[str] = None
    result_json: Optional[Any] = None
    config_json: Optional[Any] = None


class WidgetRow(BaseModel):
    id:          int
    dashboard_id: int
    widget_uid:  str
    widget_type: str
    title:       Optional[str]
    x:           int
    y:           int
    width:       int
    height:      int
    z_index:     int
    query:       Optional[str]
    sql_query:   Optional[str]
    result_json: Optional[Any]
    config_json: Optional[Any]
    created_at:  str
    updated_at:  str


class CanvasWidgetPosition(BaseModel):
    """Single widget's canvas geometry — used in bulk canvas save."""
    widget_uid: str
    x:          int
    y:          int
    width:      int
    height:     int
    z_index:    int = 1


class CanvasSave(BaseModel):
    widgets: List[CanvasWidgetPosition]


# ── Helpers ───────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_widget(row) -> dict:
    d = dict(row)
    for field in ("result_json", "config_json"):
        if d.get(field) and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (ValueError, TypeError):
                pass
    return d


def _get_dashboard_or_404(db, dashboard_id: int):
    row = db.execute(
        "SELECT * FROM dashboard WHERE id = ?", (dashboard_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Dashboard {dashboard_id} not found.")
    return row


def _get_widget_or_404(db, dashboard_id: int, widget_uid: str):
    row = db.execute(
        "SELECT * FROM dashboard_widget WHERE dashboard_id = ? AND widget_uid = ?",
        (dashboard_id, widget_uid),
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Widget '{widget_uid}' not found in dashboard {dashboard_id}.",
        )
    return row


# ── Dashboard endpoints ───────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED, response_model=DashboardRow)
def create_dashboard(body: DashboardCreate):
    """Create a new dashboard."""
    now = _now()
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO dashboard (name, description, created_at, updated_at) VALUES (?,?,?,?)",
            (body.name, body.description, now, now),
        )
        return dict(db.execute(
            "SELECT * FROM dashboard WHERE id = ?", (cur.lastrowid,)
        ).fetchone())


@router.get("", response_model=List[DashboardRow])
def list_dashboards():
    """Return all dashboards ordered newest-first."""
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM dashboard ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/{dashboard_id}")
def get_dashboard(dashboard_id: int):
    """Return a dashboard and all its widgets."""
    with get_db() as db:
        dash = _get_dashboard_or_404(db, dashboard_id)
        widgets = db.execute(
            "SELECT * FROM dashboard_widget WHERE dashboard_id = ? ORDER BY z_index",
            (dashboard_id,),
        ).fetchall()
    return {
        **dict(dash),
        "widgets": [_row_to_widget(w) for w in widgets],
    }


@router.patch("/{dashboard_id}", response_model=DashboardRow)
def update_dashboard(dashboard_id: int, body: DashboardPatch):
    """Partial update of name / description."""
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        fields, vals = [], []
        if body.name is not None:
            fields.append("name = ?"); vals.append(body.name)
        if body.description is not None:
            fields.append("description = ?"); vals.append(body.description)
        if not fields:
            raise HTTPException(status_code=422, detail="Nothing to update.")
        fields.append("updated_at = ?"); vals.append(_now())
        vals.append(dashboard_id)
        db.execute(f"UPDATE dashboard SET {', '.join(fields)} WHERE id = ?", vals)
        return dict(db.execute(
            "SELECT * FROM dashboard WHERE id = ?", (dashboard_id,)
        ).fetchone())


@router.delete("/{dashboard_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dashboard(dashboard_id: int):
    """Delete a dashboard and all its widgets (CASCADE)."""
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        db.execute("DELETE FROM dashboard WHERE id = ?", (dashboard_id,))


# ── Widget endpoints ──────────────────────────────────────────────────

@router.post(
    "/{dashboard_id}/widgets",
    status_code=status.HTTP_201_CREATED,
    response_model=WidgetRow,
)
def create_widget(dashboard_id: int, body: WidgetCreate):
    """Add a widget to a dashboard.

    Called by the frontend when the agent produces a confirmed result and
    the user places it on the canvas.  `widget_uid` is the client-generated
    UUID so the frontend can reference it immediately without round-tripping.
    """
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        now = _now()
        result_str = json.dumps(body.result_json) if body.result_json is not None else None
        config_str = json.dumps(body.config_json) if body.config_json is not None else None
        cur = db.execute(
            """
            INSERT INTO dashboard_widget
                (dashboard_id, widget_uid, widget_type, title,
                 x, y, width, height, z_index,
                 query, sql_query, result_json, config_json,
                 created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (dashboard_id, body.widget_uid, body.widget_type, body.title,
             body.x, body.y, body.width, body.height, body.z_index,
             body.query, body.sql_query, result_str, config_str, now, now),
        )
        row = db.execute(
            "SELECT * FROM dashboard_widget WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return _row_to_widget(row)


@router.get("/{dashboard_id}/widgets", response_model=List[WidgetRow])
def list_widgets(dashboard_id: int):
    """List all widgets for a dashboard, ordered by z_index."""
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        rows = db.execute(
            "SELECT * FROM dashboard_widget WHERE dashboard_id = ? ORDER BY z_index",
            (dashboard_id,),
        ).fetchall()
    return [_row_to_widget(r) for r in rows]


@router.get("/{dashboard_id}/widgets/{widget_uid}", response_model=WidgetRow)
def get_widget(dashboard_id: int, widget_uid: str):
    """Fetch a single widget."""
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        row = _get_widget_or_404(db, dashboard_id, widget_uid)
    return _row_to_widget(row)


@router.patch("/{dashboard_id}/widgets/{widget_uid}", response_model=WidgetRow)
def update_widget_endpoint(dashboard_id: int, widget_uid: str, body: WidgetPatch):
    """Partial update — handles drag (x/y), resize (width/height), z_index, etc.

    The frontend calls this on mouseup after a drag or resize so the canvas
    layout is always persisted without a full reload.
    """
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        _get_widget_or_404(db, dashboard_id, widget_uid)

        fields, vals = [], []
        simple = ("widget_type","title","x","y","width","height","z_index","query","sql_query")
        for col in simple:
            v = getattr(body, col)
            if v is not None:
                fields.append(f"{col} = ?"); vals.append(v)
        for col in ("result_json", "config_json"):
            v = getattr(body, col)
            if v is not None:
                fields.append(f"{col} = ?"); vals.append(json.dumps(v))

        if not fields:
            raise HTTPException(status_code=422, detail="Nothing to update.")

        fields.append("updated_at = ?"); vals.append(_now())
        vals += [dashboard_id, widget_uid]
        db.execute(
            f"UPDATE dashboard_widget SET {', '.join(fields)} "
            f"WHERE dashboard_id = ? AND widget_uid = ?",
            vals,
        )
        row = db.execute(
            "SELECT * FROM dashboard_widget WHERE dashboard_id = ? AND widget_uid = ?",
            (dashboard_id, widget_uid),
        ).fetchone()
    return _row_to_widget(row)


@router.delete(
    "/{dashboard_id}/widgets/{widget_uid}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_widget(dashboard_id: int, widget_uid: str):
    """Remove a widget from the canvas."""
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        _get_widget_or_404(db, dashboard_id, widget_uid)
        db.execute(
            "DELETE FROM dashboard_widget WHERE dashboard_id = ? AND widget_uid = ?",
            (dashboard_id, widget_uid),
        )


# ── Canvas bulk-save ──────────────────────────────────────────────────

@router.put("/{dashboard_id}/canvas", status_code=status.HTTP_200_OK)
def save_canvas(dashboard_id: int, body: CanvasSave):
    """Bulk-update all widget positions/sizes in one transaction.

    Called on manual save or auto-save after any drag/resize on the canvas.
    Only geometry fields are updated — query/result/config are untouched.
    Widgets not present in the payload are left unchanged.
    """
    now = _now()
    with get_db() as db:
        _get_dashboard_or_404(db, dashboard_id)
        for w in body.widgets:
            db.execute(
                """
                UPDATE dashboard_widget
                SET x=?, y=?, width=?, height=?, z_index=?, updated_at=?
                WHERE dashboard_id=? AND widget_uid=?
                """,
                (w.x, w.y, w.width, w.height, w.z_index, now, dashboard_id, w.widget_uid),
            )
    return {"saved": len(body.widgets), "dashboard_id": dashboard_id}
