@echo off
set "TASK_ENV=C:\Users\jungc\miniconda3\envs\pet-eye-gpu"
set "PATH=%TASK_ENV%\Library\bin;%TASK_ENV%\Scripts;%TASK_ENV%;%PATH%"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
"%TASK_ENV%\python.exe" "%~dp0train_all.py"
