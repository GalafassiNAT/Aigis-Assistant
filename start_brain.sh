#!/usr/bin/env bash
# Script de inicializacao do LLM Brain da Aigis no Linux (Vulkan / ROCm / Flash Attention)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${LLAMA_MODEL_PATH:-$SCRIPT_DIR/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"

# Procura binario do llama-server
LLAMA_BIN="${LLAMA_SERVER_BIN:-}"
if [ -z "$LLAMA_BIN" ]; then
    if [ -f "$HOME/Projects/llama.cpp/build/bin/llama-server" ]; then
        LLAMA_BIN="$HOME/Projects/llama.cpp/build/bin/llama-server"
    elif command -v llama-server &>/dev/null; then
        LLAMA_BIN="$(command -v llama-server)"
    else
        echo "[ERRO] llama-server nao encontrado!"
        echo "Defina LLAMA_SERVER_BIN=/caminho/para/llama-server"
        exit 1
    fi
fi

echo "============================================================"
echo " INICIANDO CEREBRO NEURAL DA AIGIS (Linux - Vulkan/Flash Attention)"
echo "============================================================"
echo "Binario: $LLAMA_BIN"
echo "Modelo:  $MODEL_PATH"
echo "Porta:   8080 | VRAM Offload: 100% (-ngl 99) | Contexto: 8192"
echo "============================================================"
echo

exec nice -n 15 "$LLAMA_BIN" \
    -m "$MODEL_PATH" \
    --port 8080 \
    -ngl 99 \
    --alias qwen-coder \
    -c 8192 \
    -np 1 \
    -fa on \
    -b 2048 \
    -ub 512
