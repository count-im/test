"""
BabyCoach RAG Ingest
====================
PDF (data/epigenetics/) + QA JSON (data/TL_소아청소년과/) 를 읽어
ChromaDB(data/chroma_db/)에 임베딩을 저장합니다.

실행:
    python -m app.rag.ingest
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Windows 콘솔 UTF-8 출력
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")
from typing import Generator

import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# ── 경로 설정 ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # app/rag/
_APP  = _HERE.parent                             # app/
_ROOT = _APP.parent                              # babycoach/
_DATA = _ROOT / "data"

PDF_DIR    = _DATA / "epigenetics"
QA_DIR     = _DATA / "TL_소아청소년과"
CHROMA_DIR = _DATA / "chroma_db"
COLLECTION = "babycoach_rag"

CHUNK_SIZE    = 500   # PDF 청킹 문자 수
CHUNK_OVERLAP = 100
EMBED_MODEL   = "text-embedding-3-small"
EMBED_BATCH   = 100   # OpenAI API 호출당 최대 텍스트 수

# ── .env 로드 ──────────────────────────────────────────────────
_DOTENV_CANDIDATES = [
    _ROOT / ".env",
    Path(r"D:\PyProject\env_keys\.env"),
]
for _p in _DOTENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p)
        break


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "[ERROR] OPENAI_API_KEY가 없습니다.\n"
            "  babycoach/.env 또는 D:\\PyProject\\env_keys\\.env 에 설정하세요."
        )
    return OpenAI(api_key=api_key)


# ── PDF 청킹 ───────────────────────────────────────────────────
def _extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    """PDF 텍스트를 CHUNK_SIZE 단위로 분할해 반환."""
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # 청킹
    chunks = []
    start = 0
    idx = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append({
                "text":        chunk,
                "source":      "pdf",
                "filename":    pdf_path.name,
                "chunk_index": idx,
            })
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks


# ── QA JSON 로드 ───────────────────────────────────────────────
def _load_qa_chunks(qa_dir: Path) -> Generator[dict, None, None]:
    """JSON 파일 1개 = 청크 1개 (question + answer 통합)."""
    files = sorted(qa_dir.glob("*.json"))
    for idx, fpath in enumerate(files):
        try:
            with open(fpath, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {fpath.name}: {e}")
            continue

        # 필드명 유연 처리 (question/Q/질문, answer/A/답변)
        q = (data.get("question") or data.get("Q") or data.get("질문") or "").strip()
        a = (data.get("answer")   or data.get("A") or data.get("답변") or "").strip()

        # 중첩 구조 처리 ({"data": {"question": ...}})
        if not q and isinstance(data.get("data"), dict):
            inner = data["data"]
            q = (inner.get("question") or inner.get("Q") or inner.get("질문") or "").strip()
            a = (inner.get("answer")   or inner.get("A") or inner.get("답변") or "").strip()

        text = f"Q: {q}\nA: {a}" if q or a else json.dumps(data, ensure_ascii=False)[:1000]
        yield {
            "text":        text,
            "source":      "qa",
            "filename":    fpath.name,
            "chunk_index": idx,
        }


# ── 임베딩 (배치) ──────────────────────────────────────────────
def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ── ChromaDB 저장 ──────────────────────────────────────────────
def _get_collection(reset: bool = False) -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            chroma.delete_collection(COLLECTION)
            print(f"[INFO] 기존 컬렉션 '{COLLECTION}' 삭제 완료")
        except Exception:
            pass
    return chroma.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _upsert_chunks(
    col: chromadb.Collection,
    client: OpenAI,
    chunks: list[dict],
    id_prefix: str,
) -> int:
    """청크 리스트를 배치로 임베딩 후 ChromaDB에 upsert."""
    total = 0
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        texts = [c["text"] for c in batch]
        embeddings = _embed_batch(client, texts)
        ids        = [f"{id_prefix}_{i + j}" for j, _ in enumerate(batch)]
        metadatas  = [
            {
                "source":      c["source"],
                "filename":    c["filename"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)
    return total


# ── 메인 ──────────────────────────────────────────────────────
def main(reset: bool = True) -> None:
    print("=" * 60)
    print("BabyCoach RAG Ingest 시작")
    print("=" * 60)

    openai_client = _get_openai_client()
    col = _get_collection(reset=reset)

    grand_total = 0

    # ── PDF ────────────────────────────────────────────────────
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"\n[PDF] {len(pdf_files)}개 파일 처리 시작")
    for pdf_path in pdf_files:
        chunks = _extract_pdf_chunks(pdf_path)
        added  = _upsert_chunks(col, openai_client, chunks, f"pdf_{pdf_path.stem}")
        grand_total += added
        print(f"  ✓ {pdf_path.name}  →  {added}개 청크")

    # ── QA JSON ───────────────────────────────────────────────
    qa_chunks = list(_load_qa_chunks(QA_DIR))
    total_qa  = len(qa_chunks)
    print(f"\n[QA] {total_qa}개 JSON 파일 임베딩 시작 (배치 {EMBED_BATCH}개씩)...")
    added = _upsert_chunks(col, openai_client, qa_chunks, "qa")
    grand_total += added
    print(f"  ✓ QA JSON  →  {added}개 청크")

    print("\n" + "=" * 60)
    print(f"완료! ChromaDB 저장 위치: {CHROMA_DIR}")
    print(f"총 청크 수: {grand_total:,}개  |  컬렉션: {COLLECTION}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BabyCoach RAG Ingest")
    parser.add_argument("--no-reset", action="store_true", help="기존 DB 유지 (추가 모드)")
    args = parser.parse_args()
    main(reset=not args.no_reset)
