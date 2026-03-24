from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ..db import get_connection
from ..graph import run_recommendation
from ..schemas import RecommendResponse
from ..state import BabyCoachState, build_state_from_input


router = APIRouter(prefix="", tags=["recommend"])


def _upsert_baby_profile(merged: Dict[str, Any]) -> int:
    """baby_profile에 이름 기준 UPSERT 후 baby_id 반환."""
    name       = str(merged.get("baby_name") or "anonymous").strip() or "anonymous"
    age_months = int(merged.get("age_months") or 0)
    weight_kg  = float(merged.get("weight_kg") or 0.0)
    allergies  = json.dumps(merged.get("allergies") or [], ensure_ascii=False)
    notes      = str(merged.get("notes") or "")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM baby_profile WHERE name = ? ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()

        if row:
            baby_id = int(row["id"])
            conn.execute(
                """
                UPDATE baby_profile
                SET age_months=?, weight_kg=?, allergies=?, notes=?
                WHERE id=?
                """,
                (age_months, weight_kg, allergies, notes, baby_id),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO baby_profile (name, age_months, weight_kg, allergies, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, age_months, weight_kg, allergies, notes),
            )
            baby_id = int(cur.lastrowid)

        conn.commit()
    return baby_id


def _log_activity(baby_id: int, merged: Dict[str, Any], final_output: Dict[str, Any]) -> None:
    """activity_log에 recommend 결과 기록."""
    nudge_obj = final_output.get("nudge") or {}
    payload = {
        "age_months": merged.get("age_months"),
        "weight_kg":  merged.get("weight_kg"),
        "allergies":  merged.get("allergies") or [],
        "spoon":      final_output.get("spoon"),
        "nudge":      nudge_obj.get("nudge_message") if isinstance(nudge_obj, dict) else nudge_obj,
    }
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO activity_log (baby_id, type, payload) VALUES (?, ?, ?)",
            (baby_id, "recommend", json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: Dict[str, Any]) -> RecommendResponse:
    """
    Run BabyCoach LangGraph and return `final_output`.

    PoC 2차 요구:
    - UI가 아래 중첩 payload를 보낼 수 있음
      { child_profile, spoon_input, play_input }
    - 기존 flat input도 호환하도록 병합 처리합니다.
    - 추천 완료 후 baby_profile UPSERT + activity_log 기록.
    """

    try:
        merged: Dict[str, Any] = {}
        if isinstance(payload, dict) and "child_profile" in payload:
            merged.update(payload.get("child_profile") or {})
            merged.update(payload.get("spoon_input") or {})
            merged.update(payload.get("play_input") or {})

            baby_info = payload.get("baby_info") or {}
            if isinstance(baby_info, dict):
                health = baby_info.get("health") or {}
                happy  = baby_info.get("happy") or {}
                growth_direction = happy.get("growth_direction") or []
                if isinstance(growth_direction, list):
                    merged["growth_direction"] = growth_direction
                if health.get("name"):
                    merged["baby_name"] = health.get("name")
                if health.get("birth_date"):
                    merged["birth_date"] = health.get("birth_date")

            if payload.get("parent_query"):
                merged["parent_query"] = payload.get("parent_query")
            elif isinstance(baby_info, dict):
                happy = baby_info.get("happy") or {}
                gd = happy.get("growth_direction") or []
                if gd:
                    merged["parent_query"] = "부모 성장 방향: " + ", ".join([str(x) for x in gd])
        else:
            merged = payload

        input_state: BabyCoachState = build_state_from_input(merged)
        final_state  = run_recommendation(input_state)
        final_output = final_state.get("final_output", {})
        if not isinstance(final_output, dict):
            raise TypeError("final_output must be a dict")

        # ── DB 저장 ─────────────────────────────────────
        baby_id: Optional[int] = None
        try:
            baby_id = _upsert_baby_profile(merged)
            _log_activity(baby_id, merged, final_output)
        except Exception:
            pass  # DB 실패가 추천 응답을 막지 않도록

        return RecommendResponse(final_output=final_output, baby_id=baby_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

