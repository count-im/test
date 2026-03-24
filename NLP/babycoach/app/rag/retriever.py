"""
BabyCoach RAG Retriever
=======================
ChromaDB에서 유사 청크를 검색합니다.

사용 예:
    from app.rag.retriever import query_rag
    results = query_rag("이유식 단백질 시작 시기", top_k=5)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── 경로 설정 ──────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent   # app/rag/
_ROOT      = _HERE.parent.parent               # babycoach/
_CHROMA    = _ROOT / "data" / "chroma_db"
COLLECTION = "babycoach_rag"
EMBED_MODEL = "text-embedding-3-small"

# ── .env 로드 ──────────────────────────────────────────────────
for _p in [_ROOT / ".env", Path(r"D:\PyProject\env_keys\.env")]:
    if _p.exists():
        load_dotenv(_p)
        break


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    if not _CHROMA.exists():
        raise RuntimeError(
            "RAG DB가 없습니다. ingest.py를 먼저 실행하세요.\n"
            f"  python -m app.rag.ingest"
        )
    chroma = chromadb.PersistentClient(path=str(_CHROMA))
    try:
        return chroma.get_collection(name=COLLECTION)
    except Exception:
        raise RuntimeError(
            f"컬렉션 '{COLLECTION}'이 없습니다. ingest.py를 먼저 실행하세요.\n"
            f"  python -m app.rag.ingest"
        )


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAI(api_key=api_key)


def _embed_query(query: str) -> list[float]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    return resp.data[0].embedding


def query_rag(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    ChromaDB에서 query와 가장 유사한 청크를 top_k개 반환합니다.

    Returns:
        list of dict:
            - text     : 청크 본문
            - source   : "pdf" | "qa"
            - filename : 원본 파일명
            - distance : 코사인 거리 (낮을수록 유사)
    """
    col       = _get_collection()
    embedding = _embed_query(query)

    results = col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict[str, Any]] = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append(
            {
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "filename": meta.get("filename", ""),
                "distance": round(dist, 4),
            }
        )
    return output


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "이유식 단백질 첫 도입 시기"
    print(f"쿼리: {q}\n")
    for i, r in enumerate(query_rag(q, top_k=3), 1):
        print(f"[{i}] source={r['source']} | file={r['filename']} | dist={r['distance']}")
        print(f"     {r['text'][:120]}...")
        print()
