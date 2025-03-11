# Use Python 3.10 slim base image
FROM python:3.10-slim

# Add metadata
LABEL maintainer="brandon.colelough@nih.gov"
LABEL description="ClinIQLink Docker image for AI model evaluation"

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt from the correct location inside model_submission/submission
COPY submission/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r /app/requirements.txt

# Download necessary NLTK models
RUN python -c "import nltk; nltk.download('punkt')"

# Set environment variable for dataset directory (Codabench format)
# This ensures that the dataset is accessed from the Codabench competition dataset mount point
ENV CODABENCH_DATASET_DIR="/data/Codabench_ClinIQLink_hidden_dataset_initial_upload"

# Copy evaluation script and required folders
COPY submission/submit.py /app/
COPY model_submission/ /app/model_submission/
COPY submission/submission_template/ /app/submission_template/

# Ensure script has execution permissions
RUN chmod +x /app/submit.py

# Default command to run the evaluation script
CMD ["python", "/app/submit.py"]
