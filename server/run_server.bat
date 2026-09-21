@echo off
set "TASK_ENV=C:\Users\jungc\miniconda3\envs\pet-eye-gpu"
set "PATH=%TASK_ENV%\Library\bin;%TASK_ENV%\Scripts;%TASK_ENV%;%PATH%"
set "PYTHONUNBUFFERED=1"
echo Pet Eye Check AI server: http://0.0.0.0:8000
"%TASK_ENV%\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
