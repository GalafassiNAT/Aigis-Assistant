@echo off
chcp 65001 >nul
title Aigis Brain Server (llama-server)

echo ============================================================
echo 🧠 INICIANDO CÉREBRO NEURAL DA AIGIS (ROCm / AMD RX 7600)
echo ============================================================
echo Modelo: aigis-qwen-7b-q4_k_m.gguf (LoRA Fine-Tuned)
echo Porta: 8080 ^| VRAM Offload: 100%% (-ngl 99) ^| Contexto: 8192
echo ============================================================
echo.

set "LLAMA_PATH=C:\Users\Natanael Galafassi\Desktop\Local LLM\llama-b11716-bin-win-rocm-10.0-x64\llama-server.exe"
set "MODEL_PATH=C:\Users\Natanael Galafassi\Desktop\Projects\AIAssistant\models\aigis-qwen-7b-q4_k_m.gguf"

"%LLAMA_PATH%" -m "%MODEL_PATH%" --port 8080 -ngl 99 --alias qwen-coder -c 8192

pause
