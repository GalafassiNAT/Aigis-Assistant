# Inicia o servidor do cérebro local da Aigis (Qwen 2.5 Coder 7B LoRA Finetuned)
$ErrorActionPreference = "Stop"

$llamaServer = "C:\Users\Natanael Galafassi\Desktop\Local LLM\llama-b11716-bin-win-rocm-10.0-x64\llama-server.exe"
$modelFile = Join-Path $PSScriptRoot "models\aigis-qwen-7b-q4_k_m.gguf"

if (-not (Test-Path $llamaServer)) {
    Write-Host "[ERRO] Executavel do llama-server nao encontrado em: $llamaServer" -ForegroundColor Red
    pause
    exit 1
}

if (-not (Test-Path $modelFile)) {
    Write-Host "[ERRO] Modelo $modelFile nao encontrado!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🧠 INICIANDO CÉREBRO NEURAL DA AIGIS (ROCm / AMD RX 7600)" -ForegroundColor Green
Write-Host "Modelo: aigis-qwen-7b-q4_k_m.gguf (LoRA Fine-Tuned)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

& $llamaServer -m $modelFile --port 8080 -ngl 99 --alias qwen-coder -c 8192
