import json

path = "모델배포개론01.ipynb"

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

modified = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])

    # venv 관련 셀 스킵 처리
    if any(kw in src for kw in ["python -m venv", ".venv/bin/", "ipykernel install"]):
        cell["source"] = ["# [SKIP] Colab 미지원 - 개념 확인용 셀, 실행하지 않습니다\n"] + cell["source"]
        modified += 1

    # --break-system-packages 제거
    new_src = []
    for line in cell["source"]:
        new_src.append(line.replace(" --break-system-packages", ""))
    cell["source"] = new_src

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"수정 완료: {modified}개 셀 처리됨")
