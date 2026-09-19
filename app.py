import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

# File Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_fraud_model.pkl')

# Jinja2 Templates setup
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load Model
model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    Amount: float
    Time: float
    V1: float
    V2: float
    V3: float

@app.get("/")
def home(request: Request):
    # Updated TemplateResponse syntax
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/predict")
def predict_fraud(data: Transaction):
    # Pydantic v2 support
    input_data = pd.DataFrame([data.model_dump()])
    
    prob = model.predict_proba(input_data)[0][1]
    is_fraud = bool(prob >= 0.4)

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(float(prob), 4),
        "status": "ALERT" if is_fraud else "APPROVED"
    }