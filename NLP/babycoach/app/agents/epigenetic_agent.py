from __future__ import annotations

from ..state import BabyCoachState


def epigenetic_agent(state: BabyCoachState) -> BabyCoachState:
    """
    Epigenetic node (state -> state).

    PoC에서는 '실제 분자/의학적 주장'이 아니라,
    생활 루틴/스트레스 완화/감각 환경을 설명하는 요약을 만듭니다.
    RAG가 활성화된 경우 관련 논문/QA 청크를 컨텍스트로 추가합니다.
    """

    meal_refusal = bool(state.get("meal_refusal", False))
    refusal      = bool(state.get("refusal", False))
    parent_note  = state.get("parent_note", "") or ""
    age_months   = state.get("age_months", 0)

    epigenetic_summary = "오늘은 '자극을 크게 늘리기'보다 '예측 가능하게'를 우선하는 흐름이에요."
    if meal_refusal or refusal:
        epigenetic_summary = (
            "식사/놀이에서 거부 신호가 있을 수 있어서, 오늘은 자극을 줄이고 속도를 낮춰 "
            "예측 가능한 루틴(짧게-반복-끝내기)을 더 강조했어요."
        )

    if parent_note.strip():
        epigenetic_summary += f" 부모 메모도 참고해서: {parent_note.strip()}"

    # ── RAG 검색 (선택적 — DB 없으면 조용히 스킵) ──────────────
    rag_context: list[dict] = []
    try:
        from ..rag.retriever import query_rag

        query = f"{age_months}개월 아기 식이 환경 epigenetic 루틴"
        if meal_refusal or refusal:
            query = f"{age_months}개월 이유식 거부 스트레스 완화 epigenetic"

        rag_context = query_rag(query, top_k=3)

        if rag_context:
            sources_text = "\n".join(
                f"  [{i+1}] ({r['source'].upper()} / {r['filename']}) {r['text'][:200]}"
                for i, r in enumerate(rag_context)
            )
            epigenetic_summary += f"\n\n[RAG 근거]\n{sources_text}"
    except RuntimeError:
        # RAG DB 미구축 상태 — 정상 동작 유지
        pass
    except Exception:
        pass

    return {
        **state,
        "epigenetic_summary": epigenetic_summary,
        "rag_context": rag_context,
    }
