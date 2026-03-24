from __future__ import annotations

import json
import urllib.request


BASE_URL = "http://127.0.0.1:8002"


def _request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        return resp.getcode(), json.loads(body)


def main() -> None:
    payload = {
        "child_profile": {
            "age_months": 11,
            "weight_kg": 8.9,
            "allergies": ["우유"],
            "notes": "테스트 메모",
        },
        "baby_info": {
            "health": {
                "name": "테스트아기",
                "birth_date": "2025-04-10",
                "allergy_custom": "갑각류",
            },
            "wisdom": {
                "likes": ["음악"],
                "traits": ["호기심 많음"],
                "dev_focus": ["언어"],
            },
            "happy": {
                "parent_hopes": ["수면 안정"],
                "current_worries": ["식사 거부"],
                "baby_status": ["예민함"],
                "growth_direction": ["정서적으로 안정된 아이", "사회성이 좋은 아이"],
                "free_text": "잘 자랐으면 좋겠어요",
            },
        },
    }

    post_status, post_body = _request("POST", "/api/baby-profile", payload)
    print("POST /api/baby-profile")
    print("status:", post_status)
    print("body:", post_body)

    get_status, get_body = _request("GET", "/api/baby-profile")
    print("\nGET /api/baby-profile")
    print("status:", get_status)
    print("body:", get_body)

    assert post_status == 200
    assert post_body.get("ok") is True
    assert get_status == 200
    assert get_body.get("has_profile") is True
    assert get_body.get("baby_info", {}).get("happy", {}).get("growth_direction"), "growth_direction missing"
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()

