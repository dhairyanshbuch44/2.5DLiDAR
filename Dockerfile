# ==============================================================================
# Fovea-LiDAR Production Docker Container
# Supports: Hugging Face Spaces, Render, Railway, Fly.io, or Local Docker
# ==============================================================================
FROM python:3.11-slim

# Prevent interactive prompts and Python output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies (build-essential, git, libgl for headless rendering if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definitions first for optimal layer caching
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application source, configs, data, and dashboard
COPY configs/ ./configs/
COPY data/ ./data/
COPY dashboard/ ./dashboard/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./

# Hugging Face Spaces and Render default port
EXPOSE 7860

# Launch FastAPI dashboard with WebSocket support
CMD ["sh", "-c", "uvicorn dashboard.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
