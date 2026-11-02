@echo off
set "TASK_ENV=C:\Users\jungc\miniconda3\envs\pet-eye-gpu"
set "PATH=%TASK_ENV%\Library\bin;%TASK_ENV%\Scripts;%TASK_ENV%;%PATH%"
echo Single-disease experiment only. Uses confirmed data and does not deploy.
"%TASK_ENV%\python.exe" "%~dp0train.py" --disease cataract --epochs 1
