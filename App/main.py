from fastapi import FastAPI, UploadFile, File
import torch
import librosa
import numpy as np
import tempfile
import os

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/wav2vec2-base-960h"

# =============================
# MODELE TRANSFORMER
# =============================
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

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths=None):
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling temporel
        return self.classifier(x)

# =============================
# CHARGEMENT DES MODELES
# =============================
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
wav2vec2.eval()

model_transformer = FatigueTransformer().to(DEVICE)
model_transformer.load_state_dict(
    torch.load("model/fatigue_transformer.pth", map_location=DEVICE)
)
model_transformer.eval()

# =============================
# FASTAPI
# =============================
app = FastAPI(title="API Détection Fatigue Conducteur")

# =============================
# FONCTION PREDICTION
# =============================
def predict(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec2(**inputs)
        embeddings = outputs.last_hidden_state  # (1, T, 768)

        logits = model_transformer(embeddings)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    classes = ["Reposée", "Fatiguée"]
    pred = classes[int(np.argmax(probs))]

    return pred, probs.tolist()

# =============================
# ROUTES API
# =============================
@app.get("/")
def root():
    return {"message": "API Détection Fatigue active 🚗"}

@app.post("/predict")
async def predict_fatigue(file: UploadFile = File(...)):
    # sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result, probabilities = predict(tmp_path)

    os.remove(tmp_path)

    return {
        "prediction": result,
        "probabilities": {
            "reposee": round(probabilities[0],2),
            "fatiguee": round(probabilities[1],2)
        }
    }
