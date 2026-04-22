FROM python:3.11-slim

# System deps for audio (pydub/ffmpeg) and PyAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create necessary directories
RUN mkdir -p data/checkpoints data/raw data/processed data/uploads logs

# Expose Flask port
EXPOSE 5000

ENV FLASK_ENV=production
ENV PORT=5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:create_app()"]
