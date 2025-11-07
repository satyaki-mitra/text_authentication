FROM python:3.10-slim

# Set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/transformers \
    HF_DATASETS_CACHE=/tmp/datasets \
    TOKENIZERS_PARALLELISM=false


# Create necessary directories
RUN mkdir -p /tmp/huggingface /tmp/transformers /tmp/datasets /app/data/reports /app/data/uploads /app/models/cache /app/logs

# Copy requirements first for better caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Clear any incompatible cached models
RUN rm -rf /tmp/huggingface/* /tmp/transformers/* /app/models/cache/*

# Expose port 7860 (hugging Face Spaces Standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1


# Run the application
CMD ["uvicorn", "text_auth_app:app", "--host", "0.0.0.0", "--port", "7860"]