#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 MODEL.gguf --name MODEL_NAME [--quant Q4_K_M] [--overwrite]"
  exit 1
fi

GGUF_INPUT="$1"
shift

MODEL_NAME=""
QUANT=""
OVERWRITE="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --quant)
      QUANT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ ! -f "$GGUF_INPUT" ]; then
  echo "Error: GGUF file not found:"
  echo "  $GGUF_INPUT"
  exit 1
fi

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="$(basename "$GGUF_INPUT" .gguf)"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama is not in PATH."
  exit 1
fi

if ! ollama list >/dev/null 2>&1; then
  echo "Error: Ollama is not running."
  echo "Start it yourself first with:"
  echo "  ollama serve"
  exit 1
fi

GGUF_DIR="$(cd -- "$(dirname -- "$GGUF_INPUT")" && pwd -P)"
GGUF_FILE="$(basename "$GGUF_INPUT")"
GGUF_PATH="$GGUF_DIR/$GGUF_FILE"
MODELFILE="$GGUF_DIR/Modelfile.$MODEL_NAME"

cat > "$MODELFILE" <<EOF
FROM $GGUF_PATH

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
EOF

if [ "$OVERWRITE" = "true" ]; then
  ollama stop "$MODEL_NAME" >/dev/null 2>&1 || true
  ollama rm "$MODEL_NAME" >/dev/null 2>&1 || true
fi

if [ -n "$QUANT" ]; then
  ollama create "$MODEL_NAME" -f "$MODELFILE" -q "$QUANT"
else
  ollama create "$MODEL_NAME" -f "$MODELFILE"
fi

echo "Run:"
echo "  ollama run $MODEL_NAME"
