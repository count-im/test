from __future__ import annotations


def baby_call_name_for_coaching(full_name: str) -> str:
    """
    코칭/대화에서 부를 이름: 성(첫 글자)을 뺀 뒤 + '이'.
    예: 전서연 -> 서연이

    - 1글자만 있으면 그대로 + 이 (예: 민 -> 민이)
    - 공백은 앞뒤만 제거
    """
    name = (full_name or "").strip()
    if not name:
        return ""
    if len(name) == 1:
        return name + "이"
    core = name[1:]
    return core + "이" if core else name + "이"


def apply_baby_name_to_coaching_text(text: str, full_name: str) -> str:
    """
    LLM 출력 등에 남은 등록명을 호칭으로 치환.
    한 글자 이름은 일반 단어와 충돌할 수 있어 2글자 이상일 때만 치환합니다.
    """
    s = text or ""
    full = (full_name or "").strip()
    if not full or not s:
        return s
    call = baby_call_name_for_coaching(full)
    if not call or call == full:
        return s
    if len(full) < 2:
        return s
    return s.replace(full, call)


def sanitize_coaching_dict(final_output: dict, full_name: str) -> dict:
    """
    /recommend final_output 내 사용자에게 보이는 문자열 필드에 호칭 규칙 적용.
    """
    full = (full_name or "").strip()
    if not full:
        return final_output

    out = dict(final_output)
    spoon = dict(out.get("spoon") or {})
    play = dict(out.get("play") or {})
    growth = dict(out.get("growth") or {})
    nudge = dict(out.get("nudge") or {})
    explanation = dict(out.get("explanation") or {})

    if isinstance(spoon.get("notes"), str):
        spoon["notes"] = apply_baby_name_to_coaching_text(spoon["notes"], full)
    sug = spoon.get("suggestions")
    if isinstance(sug, list):
        spoon["suggestions"] = [apply_baby_name_to_coaching_text(str(x), full) for x in sug]

    if isinstance(play.get("notes"), str):
        play["notes"] = apply_baby_name_to_coaching_text(play["notes"], full)
    ps = play.get("suggestions")
    if isinstance(ps, list):
        play["suggestions"] = [apply_baby_name_to_coaching_text(str(x), full) for x in ps]

    obs = growth.get("observation_points")
    if isinstance(obs, list):
        growth["observation_points"] = [apply_baby_name_to_coaching_text(str(x), full) for x in obs]

    if isinstance(nudge.get("nudge_message"), str):
        nudge["nudge_message"] = apply_baby_name_to_coaching_text(nudge["nudge_message"], full)
    if isinstance(explanation.get("explanation"), str):
        explanation["explanation"] = apply_baby_name_to_coaching_text(explanation["explanation"], full)

    out["spoon"] = spoon
    out["play"] = play
    out["growth"] = growth
    out["nudge"] = nudge
    out["explanation"] = explanation

    ccs = out.get("chat_context_summary")
    if isinstance(ccs, str):
        out["chat_context_summary"] = apply_baby_name_to_coaching_text(ccs, full)

    return out
