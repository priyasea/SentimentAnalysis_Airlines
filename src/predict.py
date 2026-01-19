# predict.py

from typing import Dict, Any
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from fastapi.responses import HTMLResponse
from pathlib import Path
 
# =========================
# Load saved model pipeline
# =========================
MODEL_PATH = "models/model.pkl"
BASE_DIR = Path(__file__).resolve().parent.parent
with open(MODEL_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)



# =========================
# FastAPI App Initialization
# =========================

# =========================
# Pydantic Input Schema
# =========================


class Sentiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    airline: Literal[
        "american",
        "delta",
        "southwest",
        "us airways",
        "united",
        "virgin america"
    ]
    retweet_count: int = Field(..., ge=0)
    
    
# -------------------------------------------------------
# Preprocess input
# -------------------------------------------------------

class SentimentPrediction(BaseModel):
    sentiment: int
    review: str

app = FastAPI(title="Airline Review Sentiment Prediction API", version="1.0")

# =========================
# Prediction Function
# =========================

def predict_Sentiment(sentiment):
    df = pd.DataFrame([sentiment])
    prob = pipeline.predict(df)
    print(prob)
    return int(prob[0])

# =========================
# API Endpoints
# =========================
@app.get("/")
def home():
    return {"message": "Welcome to Airline Sentiment Preiction API"}

#@app.get("/ui", response_class=HTMLResponse)
#def ui():
#    html_path = BASE_DIR / "templates" / "index.html"
#    return html_path.read_text()

@app.post("/predict",response_model=SentimentPrediction)
def predict_api(sentiment: Sentiment) -> SentimentPrediction:
    prob = predict_Sentiment(sentiment.model_dump())
    pred = "Positive Review" if prob == 1 else "Negative Review"
    return SentimentPrediction(
        sentiment = prob,
        review = pred
     )
# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)