# Inicia o servidor do cérebro local da Aigis (Qwen 2.5 Coder 7B LoRA Finetuned)
$ErrorActionPreference = "Stop"

$vulkanServer = "C:\Users\Natanael Galafassi\Desktop\Local LLM\llama-b11716-bin-win-vulkan-x64\llama-server.exe"
$rocmServer   = "C:\Users\Natanael Galafassi\Desktop\Local LLM\llama-b11716-bin-win-rocm-10.0-x64\llama-server.exe"
$modelFile    = Join-Path $PSScriptRoot "models\aigis-qwen-7b-q4_k_m.gguf"

if (Test-Path $vulkanServer) {
    $llamaServer = $vulkanServer
    $backendName = "Vulkan (GPU Shaders + Flash Attention - Ultrarrápido)"
    $extraArgs = @("-fa", "on", "-b", "2048", "-ub", "512")
} elseif (Test-Path $rocmServer) {
    $llamaServer = $rocmServer
    $backendName = "ROCm / HIP"
    $extraArgs = @("-b", "2048", "-ub", "512")
} else {
    Write-Host "[ERRO] Nenhum executavel do llama-server encontrado em: $vulkanServer ou $rocmServer" -ForegroundColor Red
    pause
    exit 1
}

if (-not (Test-Path $modelFile)) {
    Write-Host "[ERRO] Modelo $modelFile nao encontrado!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🧠 INICIANDO CÉREBRO NEURAL DA AIGIS ($backendName)" -ForegroundColor Green
Write-Host "Modelo: aigis-qwen-7b-q4_k_m.gguf (LoRA Fine-Tuned)" -ForegroundColor Yellow
Write-Host "Porta: 8080 | Contexto: 4096 | Offload GPU: 100% (-ngl 99)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $llamaServer -m $modelFile --port 8080 -ngl 99 --alias qwen-coder -c 4096 -np 1 @extraArgs
