import pandas as pd
from joblib import load

MODEL_PATH = "models/rf_model.joblib"
model = load(MODEL_PATH)

LABEL_MAP = {0: "malignant", 1: "benign"}

def predict_metadata(data: dict):
    df = pd.DataFrame([data])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][pred])

    return {
        "prediction": pred,
        "label": LABEL_MAP[pred],
        "probability": prob
    }
