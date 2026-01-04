from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import librosa
import numpy as np
import tempfile
import os
import logging
import traceback

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn

from opencensus.ext.azure.log_exporter import AzureLogHandler

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# -------------------------------------------------
# Logging & Application Insights
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fatigue-detection-api")

APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONN))
    logger.info("✅ Application Insights connecté")
else:
    logger.warning("⚠️ Application Insights non configuré")

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/wav2vec2-base-960h"

logger.info(f"Appareil utilisé : {DEVICE}")

# -------------------------------------------------
# Modèle Transformer
# -------------------------------------------------
class FatigueTransformer(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths=None):
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# -------------------------------------------------
# Variables Globales
# -------------------------------------------------
processor = None
wav2vec2 = None
model_transformer = None

# -------------------------------------------------
# Initialisation FastAPI
# -------------------------------------------------
app = FastAPI(
    title="API Détection Fatigue Conducteur",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Chargement des modèles
# -------------------------------------------------
@app.on_event("startup")
async def load_models():
    global processor, wav2vec2, model_transformer
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
        wav2vec2.eval()

        model_transformer = FatigueTransformer().to(DEVICE)
        model_path = os.getenv("MODEL_PATH", "model/fatigue_transformer.pth")
        model_transformer.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        model_transformer.eval()
        
        logger.info(f"✅ Modèles chargés depuis {model_path}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèles : {e}")
        logger.error(traceback.format_exc())
        model_transformer = None

# -------------------------------------------------
# Endpoints généraux
# -------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """Route racine"""
    return {"message": "API Détection Fatigue Conducteur active 🚗"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Endpoint de santé"""
    if model_transformer is None:
        logger.warning("⚠️ Health check : Modèle non chargé")
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    logger.info("✅ Health check : OK")
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE
    }

# -------------------------------------------------
# Prédiction
# -------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fatigue(file: UploadFile = File(...)):
    """Prédiction de fatigue conducteur à partir d'un fichier audio"""
    
    if model_transformer is None:
        logger.error("❌ Prédiction : Modèle indisponible")
        raise HTTPException(status_code=503, detail="Modèle indisponible")

    file_name = file.filename
    file_size = 0

    try:
        # Sauvegarde temporaire du fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            file_size = len(content)
            tmp.write(content)
            tmp_path = tmp.name

        # Chargement et traitement audio
        audio, _ = librosa.load(tmp_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Inférence
        with torch.no_grad():
            outputs = wav2vec2(**inputs)
            embeddings = outputs.last_hidden_state
            logits = model_transformer(embeddings)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Nettoyage
        os.remove(tmp_path)

        # Résultats
        classes = ["Reposée", "Fatiguée"]
        prediction = classes[int(np.argmax(probs))]
        fatigue_prob = float(probs[1])
        repose_prob = float(probs[0])

        # Calcul du niveau de risque
        risk_level = "Low" if fatigue_prob < 0.3 else "Medium" if fatigue_prob < 0.7 else "High"

        # Logging dans Application Insights
        logger.info(
            "prediction_success",
            extra={
                "custom_dimensions": {
                    "event_type": "prediction",
                    "file_name": file_name,
                    "file_size_bytes": file_size,
                    "prediction": prediction,
                    "fatigue_probability": round(fatigue_prob, 4),
                    "repose_probability": round(repose_prob, 4),
                    "risk_level": risk_level
                }
            }
        )

        return {
            "prediction": prediction,
            "probabilities": {
                "reposee": round(repose_prob, 2),
                "fatiguee": round(fatigue_prob, 2)
            },
            "risk_level": risk_level
        }

    except ValueError as e:
        logger.error(
            f"❌ Erreur audio : {e}",
            extra={
                "custom_dimensions": {
                    "event_type": "prediction_error",
                    "error_type": "audio_processing",
                    "file_name": file_name,
                    "file_size_bytes": file_size
                }
            }
        )
        raise HTTPException(status_code=400, detail=f"Erreur traitement audio : {str(e)}")

    except Exception as e:
        logger.error(
            f"❌ Erreur prédiction : {e}",
            extra={
                "custom_dimensions": {
                    "event_type": "prediction_error",
                    "error_type": "prediction_failed",
                    "file_name": file_name,
                    "error_message": str(e)
                }
            }
        )
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erreur interne serveur")

# -------------------------------------------------
# Monitoring - Drift Detection (optionnel)
# -------------------------------------------------
@app.post("/drift/check", tags=["Monitoring"])
def check_drift(threshold: float = 0.05):
    """Détection de drift dans les données de production"""
    try:
        # Placeholder : implémenter detect_drift selon vos besoins
        logger.info(
            "drift_detection",
            extra={
                "custom_dimensions": {
                    "event_type": "drift_detection",
                    "threshold": threshold,
                    "status": "check_performed"
                }
            }
        )

        return {
            "status": "success",
            "threshold": threshold,
            "message": "Drift detection endpoint disponible"
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        raise HTTPException(status_code=500, detail="Erreur drift detection")