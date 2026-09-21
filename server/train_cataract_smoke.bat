@echo off
call "C:\Users\jungc\miniconda3\condabin\conda.bat" run -n pet-eye-gpu python "%~dp0train.py" --disease cataract --smoke --epochs 1
