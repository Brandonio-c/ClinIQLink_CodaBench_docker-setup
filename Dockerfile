# -----------------------------------------------------------------------------
# Base: CUDA 12.1 / cuDNN 8 / PyTorch 2.1.2 / Python 3.10 (with nvcc)
# -----------------------------------------------------------------------------
    FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

LABEL maintainer="brandon.colelough@nih.gov"
LABEL description="ClinIQLink Docker image for AI model submission and evaluation"

# so pip/setuptools can find CUDA
ENV CUDA_HOME=/usr/local/cuda

# -----------------------------------------------------------------------------
# 1. Non-interactive APT
# -----------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# -----------------------------------------------------------------------------
# 2. System packages
# -----------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        libopenmpi-dev \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 3. Working directory & caches
# -----------------------------------------------------------------------------
WORKDIR /app

ENV NLTK_DATA=/usr/local/nltk_data \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    TORCH_HOME=/app/.cache/torch \
    MODEL_DIR=/models \
    DATA_DIR=/data \
    OUTPUT_DIR=/results \
    USE_INTERNAL_MODEL=1

RUN mkdir -p \
        $MODEL_DIR $DATA_DIR $OUTPUT_DIR $NLTK_DATA \
        /app/results \
        /app/.cache/huggingface /app/.cache/transformers /app/.cache/torch \
        /app/models && \
    chmod -R a+rwX /app $MODEL_DIR $DATA_DIR $OUTPUT_DIR $NLTK_DATA

# -----------------------------------------------------------------------------
# 4. GPU‑specific Python wheels — **built for CUDA 12.1**
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    bitsandbytes==0.41.1 \
    faiss-gpu==1.7.2         # official cu121 wheel

# -----------------------------------------------------------------------------
# 5. Pure‑Python project dependencies
# -----------------------------------------------------------------------------
# Install Flash-Attention from the pre-built cu121 wheel
RUN pip install --no-cache-dir \
flash-attn==2.3.6 \
-f https://flash-attn-builder.s3.us-east-2.amazonaws.com/whl/cu121/torch2.1/index.html


COPY submission/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------------------------------------------------------
# 6. Cache small NLP assets
# -----------------------------------------------------------------------------
RUN python -m nltk.downloader punkt punkt_tab -d "$NLTK_DATA" && \
python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    cache_folder='/app/models'
)
EOF

# -----------------------------------------------------------------------------
# 7. Copy submission code
# -----------------------------------------------------------------------------
COPY submission/submit.py            /app/submit.py
COPY submission/evaluate.py          /app/evaluate.py
COPY submission/submission_template/ /app/submission_template/
COPY submission/entrypoint.sh        /app/entrypoint.sh

# -----------------------------------------------------------------------------
# 8. Entrypoint
# -----------------------------------------------------------------------------
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
    