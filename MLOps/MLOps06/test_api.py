import requests
import time

API_BASE = "http://localhost:8000"
VALID_KEY = "test-api-key-1234"
INVALID_KEY = "wrong-key"

# ─────────────────────────────────────────
# 테스트 1: 인증
# ─────────────────────────────────────────
print("=" * 50)
print("[테스트 1] 인증")
print("=" * 50)

r1 = requests.post(
    f"{API_BASE}/chat",
    json={"messages": [{"role": "user", "content": "안녕"}]},
    headers={"X-API-Key": VALID_KEY},
    timeout=60
)
print(f"  정상 키 → HTTP {r1.status_code}")

r2 = requests.post(
    f"{API_BASE}/chat",
    json={"messages": [{"role": "user", "content": "안녕"}]},
    headers={"X-API-Key": INVALID_KEY},
    timeout=10
)
print(f"  잘못된 키 → HTTP {r2.status_code} | {r2.json()}")

# ─────────────────────────────────────────
# 테스트 2: 멀티턴 대화
# ─────────────────────────────────────────
print()
print("=" * 50)
print("[테스트 2] 멀티턴 대화")
print("=" * 50)

messages = []
turns = ["오늘 날씨가 좋네요", "그럼 산책이라도 할까요"]

for user_msg in turns:
    messages.append({"role": "user", "content": user_msg})
    r = requests.post(
        f"{API_BASE}/chat",
        json={"messages": messages, "max_new_tokens": 50},
        headers={"X-API-Key": VALID_KEY},
        timeout=60
    )
    reply = r.json().get("reply", "")
    turn_count = r.json().get("turn_count", "")
    print(f"  턴 {turn_count} | 사용자: {user_msg}")
    print(f"         | 봇:    {reply[:60]}...")
    messages.append({"role": "assistant", "content": reply})

# ─────────────────────────────────────────
# 테스트 3: 입력 검증
# ─────────────────────────────────────────
print()
print("=" * 50)
print("[테스트 3] 입력 검증")
print("=" * 50)

# 빈 messages 전송
r3 = requests.post(
    f"{API_BASE}/chat",
    json={"messages": [], "max_new_tokens": 50},
    headers={"X-API-Key": VALID_KEY},
    timeout=10
)
print(f"  빈 messages → HTTP {r3.status_code}")

# max_new_tokens 범위 초과
r4 = requests.post(
    f"{API_BASE}/chat",
    json={"messages": [{"role": "user", "content": "테스트"}], "max_new_tokens": 9999},
    headers={"X-API-Key": VALID_KEY},
    timeout=10
)
print(f"  max_new_tokens=9999 → HTTP {r4.status_code}")

# ─────────────────────────────────────────
# 테스트 4: 동시 요청
# ─────────────────────────────────────────
print()
print("=" * 50)
print("[테스트 4] 동시 요청 (4개)")
print("=" * 50)

from concurrent.futures import ThreadPoolExecutor, as_completed

def send_chat(i):
    start = time.time()
    resp = requests.post(
        f"{API_BASE}/chat",
        json={"messages": [{"role": "user", "content": f"질문 {i+1}번입니다"}], "max_new_tokens": 30},
        headers={"X-API-Key": VALID_KEY},
        timeout=120
    )
    return {"id": i + 1, "elapsed": round(time.time() - start, 1), "status": resp.status_code}

total_start = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(send_chat, i) for i in range(4)]
    results = [f.result() for f in as_completed(futures)]

total = round(time.time() - total_start, 1)
for r in sorted(results, key=lambda x: x["id"]):
    print(f"  요청 #{r['id']}: {r['elapsed']}초 (HTTP {r['status']})")
print(f"  총 소요 시간: {total}초")

print()
print("✅ 전체 테스트 완료")
