FROM python:3.11-slim

# System deps for OpenCV + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create weights directory (populated at runtime from GCS)
RUN mkdir -p weights

# Non-root user
RUN useradd -m -u 1000 miba && chown -R miba:miba /app
USER miba

# Cloud Run expects PORT env var
ENV PORT=8080
EXPOSE 8080

# Startup: optionally download weights from GCS before serving
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
