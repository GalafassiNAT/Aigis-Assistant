@echo off
chcp 65001 >nul
title Aigis Brain Server (llama-server)

echo ============================================================
echo INICIANDO CEREBRO NEURAL DA AIGIS (Vulkan / Flash Attention)
echo ============================================================
echo Modelo: aigis-qwen-7b-q4_k_m.gguf (LoRA Fine-Tuned)
echo Porta: 8080 ^| VRAM Offload: 100%% (-ngl 99) ^| Contexto: 4096
echo ============================================================
echo.

set "VULKAN_PATH=C:\Users\Natanael Galafassi\Desktop\Local LLM\llama-b11716-bin-win-vulkan-x64\llama-server.exe"
set "ROCM_PATH=C:\Users\Natanael Galafassi\Desktop\Local LLM\llama-b11716-bin-win-rocm-10.0-x64\llama-server.exe"
set "MODEL_PATH=%~dp0models\aigis-qwen-7b-q4_k_m.gguf"

if exist "%VULKAN_PATH%" (
    "%VULKAN_PATH%" -m "%MODEL_PATH%" --port 8080 -ngl 99 --alias qwen-coder -c 4096 -np 1 -fa on -b 2048 -ub 512
) else (
    "%ROCM_PATH%" -m "%MODEL_PATH%" --port 8080 -ngl 99 --alias qwen-coder -c 4096 -np 1 -b 2048 -ub 512
)

pause
