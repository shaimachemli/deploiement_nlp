FROM python:3.10-slim-bullseye

# Désactiver cache pip (plus stable)
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dépendances système audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /App

COPY requirements.txt .

# ⚠️ Important pour PyTorch
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY App ./App
COPY model ./model 

EXPOSE 8000

CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8000"]
