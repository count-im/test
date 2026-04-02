import json, torch, numpy as np
import torch.nn as nn

class HousingModel(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.network(x)

class HousingPredictor:
    def __init__(self, model_path="models/housing_model.pth",
                 params_path="models/housing_preprocessing.json"):
        with open(params_path) as f:
            params = json.load(f)
        self.mean = np.array(params["mean"])
        self.std  = np.array(params["std"])
        self.feature_names = params["feature_names"]
        self.model = HousingModel()
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, features: list[float]) -> dict:
        x = np.array(features, dtype=np.float32)
        x_norm = (x - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(x_norm).unsqueeze(0))
        price = pred.item() * 100_000
        return {"predicted_price": round(price, 0),
                "predicted_price_unit": "USD",
                "confidence_note": "±$50,000 예상 오차"}
