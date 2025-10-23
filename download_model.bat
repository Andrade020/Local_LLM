@echo off
echo ============================================
echo  Download de Modelos GGUF
echo ============================================
echo.

REM Ativar ambiente virtual
call venv\Scripts\activate

REM Criar pasta de modelos
if not exist "models" mkdir models

echo.
echo Escolha um modelo para baixar:
echo.
echo 1. Llama-2-7B Chat Q4 (3.8 GB) - RECOMENDADO
echo 2. Mistral-7B Instruct Q4 (4.1 GB) - MUITO BOM
echo 3. Phi-2 Q4 (1.6 GB) - PEQUENO E RAPIDO
echo 4. Llama-2-13B Chat Q4 (7.4 GB) - PRECISA 16GB RAM
echo.
set /p choice="Digite o numero (1-4): "

if "%choice%"=="1" (
    echo.
    echo Baixando Llama-2-7B Chat Q4...
    hf download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False
    set MODEL_FILE=llama-2-7b-chat.Q4_K_M.gguf
)

if "%choice%"=="2" (
    echo.
    echo Baixando Mistral-7B Instruct Q4...
    hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False
    set MODEL_FILE=mistral-7b-instruct-v0.2.Q4_K_M.gguf
)

if "%choice%"=="3" (
    echo.
    echo Baixando Phi-2 Q4...
    hf download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False
    set MODEL_FILE=phi-2.Q4_K_M.gguf
)

if "%choice%"=="4" (
    echo.
    echo Baixando Llama-2-13B Chat Q4...
    hf download TheBloke/Llama-2-13B-Chat-GGUF llama-2-13b-chat.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False
    set MODEL_FILE=llama-2-13b-chat.Q4_K_M.gguf
)

if "%MODEL_FILE%"=="" (
    echo Opcao invalida!
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Download concluido!
echo ============================================
echo.
echo Arquivo: models\%MODEL_FILE%
echo.
echo Proximos passos:
echo 1. Edite config.env
echo 2. Configure MODEL_PATH=models/%MODEL_FILE%
echo 3. Execute: python main.py
echo.
pause