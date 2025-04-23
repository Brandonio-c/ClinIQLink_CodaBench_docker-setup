#!/usr/bin/env bash
set -e

MODE=${MODE:-submit}

case "$MODE" in
    submit)
        echo "Running submission pipeline..."
        python /app/submit.py \
        --mode container \
        --max_length "${MAX_LENGTH:-1028}" \
        ${EXTRA_SUBMIT_ARGS:-}
        ;;
    evaluate)
        echo "Running evaluation pipeline..."
        python /app/evaluate.py \
        --mode container \
        --results_dir "${OUTPUT_DIR:-/results}" \
        ${EXTRA_EVAL_ARGS:-}
        ;;
    *)
    echo "Unknown MODE: $MODE (expected 'submit' or 'evaluate')" >&2
    exit 1
    ;;
esac
