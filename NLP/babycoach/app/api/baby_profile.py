from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..db import get_connection


router = APIRouter(prefix="/api", tags=["baby-profile"])


def _loads_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


@router.post("/baby-profile")
def save_baby_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        child_profile = payload.get("child_profile") or {}
        baby_info = payload.get("baby_info") or {}
        health = baby_info.get("health") or {}
        wisdom = baby_info.get("wisdom") or {}
        happy = baby_info.get("happy") or {}

        name = health.get("name") or ""
        birth_date = health.get("birth_date") or ""
        baby_photo_name = health.get("baby_photo_name") or ""
        baby_photo_url = health.get("baby_photo_url") or ""
        age_months = int(child_profile.get("age_months", 0) or 0)
        weight_kg = float(child_profile.get("weight_kg", 0.0) or 0.0)

        allergies = list(child_profile.get("allergies") or [])
        custom_allergy = (health.get("allergy_custom") or "").strip()
        if custom_allergy:
            allergies.append(custom_allergy)

        notes = child_profile.get("notes") or ""
        growth_direction = happy.get("growth_direction") or []

        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO baby_profile
                    (name, birth_date, baby_photo_name, baby_photo_url, age_months, weight_kg, allergies, notes)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    birth_date,
                    baby_photo_name,
                    baby_photo_url,
                    age_months,
                    weight_kg,
                    json.dumps(allergies, ensure_ascii=False),
                    notes,
                ),
            )
            baby_id = int(cur.lastrowid)

            conn.execute(
                """
                INSERT INTO baby_context
                    (baby_id, wisdom, happiness, growth_direction)
                VALUES
                    (?, ?, ?, ?)
                """,
                (
                    baby_id,
                    json.dumps(wisdom, ensure_ascii=False),
                    json.dumps(happy, ensure_ascii=False),
                    json.dumps(growth_direction, ensure_ascii=False),
                ),
            )
            conn.commit()

        return {"ok": True, "baby_id": baby_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baby-profile")
def get_latest_baby_profile() -> Dict[str, Any]:
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    p.id AS baby_id,
                    p.name,
                    p.birth_date,
                    p.baby_photo_name,
                    p.baby_photo_url,
                    p.age_months,
                    p.weight_kg,
                    p.allergies,
                    p.notes,
                    p.created_at,
                    c.wisdom,
                    c.happiness,
                    c.growth_direction
                FROM baby_profile p
                LEFT JOIN baby_context c ON c.baby_id = p.id
                ORDER BY p.id DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return {"ok": True, "has_profile": False}

        allergies = _loads_json(row["allergies"], [])
        wisdom = _loads_json(row["wisdom"], {})
        happiness = _loads_json(row["happiness"], {})
        growth_direction = _loads_json(row["growth_direction"], [])
        if isinstance(happiness, dict) and "growth_direction" not in happiness:
            happiness["growth_direction"] = growth_direction

        child_profile = {
            "age_months": row["age_months"] or 0,
            "weight_kg": row["weight_kg"] or 0.0,
            "allergies": allergies,
            "notes": row["notes"] or "",
        }
        baby_info = {
            "health": {
                "name": row["name"] or "",
                "birth_date": row["birth_date"] or "",
                "baby_photo_name": row["baby_photo_name"] or "",
                "baby_photo_url": row["baby_photo_url"] or "",
                "allergy_custom": "",
            },
            "wisdom": wisdom if isinstance(wisdom, dict) else {},
            "happy": happiness if isinstance(happiness, dict) else {},
        }

        return {
            "ok": True,
            "has_profile": True,
            "baby_id": int(row["baby_id"]),
            "created_at": row["created_at"],
            "child_profile": child_profile,
            "baby_info": baby_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

