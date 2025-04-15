# Use Python 3.10 slim base image
FROM python:3.10-slim

# Metadata
LABEL maintainer="brandon.colelough@nih.gov"
LABEL description="ClinIQLink Docker image for AI model evaluation"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set cache paths
ENV NLTK_DATA=/usr/local/nltk_data
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch

# Copy requirements first to leverage Docker caching
COPY submission/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r /app/requirements.txt

# Download and cache NLTK 'punkt' and 'punkt_tab' offline
RUN mkdir -p $NLTK_DATA && \
    python -m nltk.downloader punkt punkt_tab -d $NLTK_DATA


# Download and cache the sentence-transformers model locally
RUN mkdir -p /app/models && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/models')"

# Copy the evaluation script and submission files
COPY submission/submit.py /app/
COPY model_submission/ /app/model_submission/
COPY submission/submission_template/ /app/submission_template/

# Permissions for Apptainer/HPC compatibility
RUN chmod -R a+rwX /app /usr/local/nltk_data

# Set dataset environment variable
ENV DATA_DIR="/data"

# Default execution command
CMD ["python", "/app/submit.py"]
