@echo off
REM Abre o LocalLLM usando o ambiente virtual (crie antes: python -m venv venv  e  venv\Scripts\pip install -r requirements.txt)
cd /d "%~dp0"
venv\Scripts\python.exe main.py %*
if errorlevel 1 pause
