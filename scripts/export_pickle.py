import pickle
from pathlib import Path

import joblib

model_dir = Path("models/production")

# Load existing trained model
model = joblib.load(model_dir / "model.joblib")

# Save the same model in pickle format
with open(model_dir / "model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Pickle model saved:", model_dir / "model.pkl")