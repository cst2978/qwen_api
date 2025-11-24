FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list
COPY requirements.txt ./

# Install Python deps
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose FastAPI (8000)
EXPOSE 8000

# Default command: FastAPI endpoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

