@echo off
chcp 65001 >nul
title Renderizar Dataset da Aigis (RVC Distillation)

echo ============================================================
echo 🎙️ RENDERIZANDO DATASET AIGIS PARA DESTILAÇÃO PIPER/VITS
echo ============================================================
echo.

set "ENV_PYTHON=C:\Users\Natanael Galafassi\micromamba\envs\ai-assistant\python.exe"

if not exist "%ENV_PYTHON%" (
    echo [AVISO] Python do ambiente ai-assistant nao encontrado no caminho padrao.
    echo Tentando python do PATH atual...
    python "%~dp02_synthesize_teacher_rvc.py"
) else (
    "%ENV_PYTHON%" "%~dp02_synthesize_teacher_rvc.py"
)

echo.
pause
