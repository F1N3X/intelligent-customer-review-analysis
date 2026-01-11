from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Charger le meilleur modèle et les objets nécessaires

MODEL_PATH = 'model_files/best_model.joblib'
VECTORIZER_PATH = 'model_files/tfidf_vectorizer.joblib'
ENCODER_PATH = 'model_files/label_encoder.joblib'

print(f"[INFO] Vérification des fichiers de modèle...")
print(f"[INFO] MODEL_PATH existe: {os.path.exists(MODEL_PATH)}")
print(f"[INFO] VECTORIZER_PATH existe: {os.path.exists(VECTORIZER_PATH)}")
print(f"[INFO] ENCODER_PATH existe: {os.path.exists(ENCODER_PATH)}")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Dummy fallback if not trained yet
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[INFO] Modèle chargé avec succès.")
    else:
        print("[ERREUR] Fichier modèle manquant.")
        model = None
except Exception as e:
    print(f"[ERREUR] Chargement du modèle: {e}")
    model = None

try:
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("[INFO] Vectorizer chargé avec succès.")
    else:
        print("[ERREUR] Fichier vectorizer manquant.")
        vectorizer = None
except Exception as e:
    print(f"[ERREUR] Chargement du vectorizer: {e}")
    vectorizer = None

try:
    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)
        print("[INFO] Label encoder chargé avec succès.")
    else:
        print("[ERREUR] Fichier label encoder manquant.")
        label_encoder = None
except Exception as e:
    print(f"[ERREUR] Chargement du label encoder: {e}")
    label_encoder = None

class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_sentiment(req: TextRequest):
    if not model or not vectorizer or not label_encoder:
        return JSONResponse({"error": "Model not available."}, status_code=500)
    X = vectorizer.transform([req.text])
    pred = model.predict(X)
    sentiment = label_encoder.inverse_transform(pred)[0]
    return {"sentiment": sentiment}
