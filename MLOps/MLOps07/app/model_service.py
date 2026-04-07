from transformers import pipeline


def load_model():
    return pipeline("text-classification", model="snunlp/KR-FinBert-SC")


def predict(model, text: str) -> dict:
    result = model(text)[0]
    label_map = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
    label = label_map.get(result["label"], result["label"])
    return {"label": label, "score": round(result["score"], 4)}
