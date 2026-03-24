from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..db import get_connection


router = APIRouter(prefix="/api", tags=["activity"])


@router.post("/activity")
def save_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        activity_type = str(payload.get("type") or "").strip().lower()
        if activity_type not in {"play", "spoon"}:
            raise ValueError("type must be one of: play, spoon")

        data = payload.get("data") or {}
        with get_connection() as conn:
            latest = conn.execute("SELECT id FROM baby_profile ORDER BY id DESC LIMIT 1").fetchone()
            baby_id = int(latest["id"]) if latest else None
            cur = conn.execute(
                """
                INSERT INTO activity_log (baby_id, type, payload)
                VALUES (?, ?, ?)
                """,
                (baby_id, activity_type, json.dumps(data, ensure_ascii=False)),
            )
            conn.commit()
            activity_id = int(cur.lastrowid)

        return {"ok": True, "activity_id": activity_id, "baby_id": baby_id, "type": activity_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

