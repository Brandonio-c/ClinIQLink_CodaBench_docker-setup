# Use Python 3.10 slim base image
FROM python:3.10-slim

# Metadata
LABEL maintainer="brandon.colelough@nih.gov"
LABEL description="ClinIQLink Docker image for AI model submission and evaluation"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Environment variables for cache and expected external paths
ENV NLTK_DATA=/usr/local/nltk_data
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch

ENV MODEL_DIR=/models
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/results
ENV USE_INTERNAL_MODEL=1

# Create and permission all expected dirs
RUN mkdir -p $MODEL_DIR $DATA_DIR $OUTPUT_DIR $NLTK_DATA /app/results \
    && chmod -R a+rwX /app $MODEL_DIR $DATA_DIR $OUTPUT_DIR $NLTK_DATA

# Python dependencies
COPY submission/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r /app/requirements.txt

# Cache NLTK and SentenceTransformer model
RUN python -m nltk.downloader punkt punkt_tab -d $NLTK_DATA && \
    python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/models')
EOF

# Copy submission and evaluation scripts
COPY submission/submit.py /app/submit.py
COPY submission/evaluate.py /app/evaluate.py
COPY submission/submission_template/ /app/submission_template/

# Add entrypoint to switch between submit / evaluate
COPY submission/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default command (handled by entrypoint)
ENTRYPOINT ["/app/entrypoint.sh"]
