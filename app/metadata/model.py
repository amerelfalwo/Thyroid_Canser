import pandas as pd
from joblib import load

MODEL_PATH = "models/rf_model.joblib"
model = load(MODEL_PATH)

def predict_metadata(data: dict):
    df = pd.DataFrame([data])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][0])

    return {
        "prediction": pred,
        "probability_Malignant": prob
    }
