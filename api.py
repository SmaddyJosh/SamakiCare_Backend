"""
Fish Disease Detection — FastAPI Backend
Serves the trained Keras model as a REST API.

Run:
    pip install fastapi uvicorn pillow tensorflow python-multipart
    uvicorn api:app --reload --port 8000
"""

import json, io, os, random
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List


MODEL_PATH       = "model_output/best_model.keras"
CLASS_NAMES_PATH = "model_output/class_names.json"
IMAGE_SIZE       = (224, 224)
CONFIDENCE_THRESHOLD = 0.55

HEALTHY_KEYWORDS = {"healthy", "normal", "fresh"}

print("Attempting to load model …")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = json.load(f)
    print(f" Real Model loaded. Classes: {CLASS_NAMES}")
    MOCK_MODE = False
except Exception as e:
    print(f" WARNING: Real model not found. Entering MOCK MODE for frontend testing!")
    model = None
    CLASS_NAMES = ["healthy_fish", "columnaris", "bacterial_red_spot", "parasitic_lice"]
    MOCK_MODE = True


app = FastAPI(title="Fish Disease Detector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassProbability(BaseModel):
    label: str
    confidence: float
    percentage: str

class PredictionResponse(BaseModel):
    top_class: str
    is_healthy: bool
    confidence: float
    confidence_label: str   
    message: str
    all_predictions: List[ClassProbability]
    status: str             

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def confidence_label(conf: float) -> str:
    if conf >= 0.80: return "HIGH"
    if conf >= 0.55: return "MEDIUM"
    return "LOW"

@app.post("/predict", response_model=PredictionResponse)
@app.post("//predict", response_model=PredictionResponse, include_in_schema=False)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    data = await file.read()
    
  
    if MOCK_MODE:
        # Randomly decide if it's healthy or sick for testing the React UI
        top_cls = random.choice(CLASS_NAMES)
        top_conf = random.uniform(0.70, 0.98)
        
        all_preds = [
            {"label": top_cls, "confidence": top_conf},
            {"label": random.choice([c for c in CLASS_NAMES if c != top_cls]), "confidence": 1.0 - top_conf}
        ]
    else:
        
        try:
            arr = preprocess(data)
        except Exception as e:
            raise HTTPException(422, f"Could not process image: {e}")
        
        probs = model.predict(arr, verbose=0)[0]
        all_preds = sorted(
            [{"label": cls, "confidence": float(p)} for cls, p in zip(CLASS_NAMES, probs)],
            key=lambda x: x["confidence"], reverse=True
        )
        top_cls  = all_preds[0]["label"]
        top_conf = all_preds[0]["confidence"]

   
    is_healthy = any(kw in top_cls.lower() for kw in HEALTHY_KEYWORDS)
    low_conf   = top_conf < CONFIDENCE_THRESHOLD

    # Sum all the probabilities of the 'diseased' classes combined
    disease_prob = sum(p["confidence"] for p in all_preds if not any(kw in p["label"].lower() for kw in HEALTHY_KEYWORDS))

    if low_conf:
        status, message = "uncertain", f"The model is not confident enough (only {top_conf*100:.1f}%). Please upload a clearer photo."
    elif is_healthy and disease_prob >= 0.35:
        status, message = "uncertain", f"⚠️ Fish looks mostly healthy but has suspicious disease traits (combined {disease_prob*100:.1f}% probability of disease). Inspect closely!"
    elif is_healthy:
        status, message = "healthy", f"✅ The fish appears healthy ({top_conf*100:.1f}% confidence)."
    else:
        clean = top_cls.replace("_", " ").title()
        status, message = "diseased", f"⚠️ Possible condition detected: {clean} ({top_conf*100:.1f}% confidence). Consult an aquatic vet."

    return PredictionResponse(
        top_class=top_cls, is_healthy=is_healthy, confidence=round(top_conf, 4),
        confidence_label=confidence_label(top_conf), message=message, status=status,
        all_predictions=[
            ClassProbability(label=p["label"], confidence=round(p["confidence"], 4), percentage=f"{p['confidence']*100:.1f}%")
            for p in all_preds
        ],
    )