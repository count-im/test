import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0)
X_train_norm = (X_train - train_mean) / train_std
X_test_norm  = (X_test  - train_mean) / train_std

X_train_t = torch.FloatTensor(X_train_norm)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t  = torch.FloatTensor(X_test_norm)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

model = HousingModel()
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(1, 51):
    for Xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/50 — Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = model(X_test_t)
    rmse = torch.sqrt(criterion(preds, y_test_t)).item()
print(f"\nTest RMSE: {rmse:.4f} ($100K 단위)")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/housing_model.pth")
with open("models/housing_preprocessing.json", "w") as f:
    json.dump({"mean": train_mean.tolist(), "std": train_std.tolist(),
               "feature_names": feature_names}, f, indent=2)
print("✅ 모델 저장 완료: models/")
