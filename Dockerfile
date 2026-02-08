# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /app

# Dipendenze di build per creare wheel (solo nel builder)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc libffi-dev libssl-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir=/wheels -r requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Solo runtime deps necessari (curl per l'installer di Ollama, zstd per decompressione)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl zstd ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copia le wheel prodotte e installa senza accedere a PyPI
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Installa Ollama (rimuove eventuali temp dell'installer)
RUN curl -fsSL https://ollama.com/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

# App
COPY . .
RUN chmod +x run-docker.sh

EXPOSE 8080
ENTRYPOINT ["./run-docker.sh"]
