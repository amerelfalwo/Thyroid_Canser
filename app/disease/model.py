import pandas as pd
from joblib import load

MODEL_PATH = "models/disease.pkl"
model = load(MODEL_PATH)

LABEL_MAP = {0: 'normal', 1: 'hypothyroid', 2: 'hyperthyroid'}

def predict_thyroid(data: dict):
    df = pd.DataFrame([data])
    pred = int(model.predict(df)[0])
    probs = model.predict_proba(df)[0]
    prob_dict = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}
    
    return {
        "prediction": pred,
        "label": LABEL_MAP[pred],
        "probabilities": prob_dict
    }
