# Dockerfile
FROM python:3.12-slim

# 1) Basic Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 2) System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3) Install Python dependencies
COPY requirements.inference.txt .
RUN pip install --no-cache-dir -r requirements.inference.txt

# 4) Copy code and artifacts
COPY src ./src
COPY models ./models
COPY pyproject.toml .

# 5) Install the package in editable mode for the imports to work correctly
RUN pip install --no-cache-dir -e .

# 6) Expose the port for FastAPI
EXPOSE 8010

# 7) Execute the FastAPI application using Uvicorn
CMD ["python", "-m", "uvicorn", "credit_ml.api.main:app", "--host", "0.0.0.0", "--port", "8010"]

# 8) Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health').read()"