@echo off
call "C:\Users\jungc\miniconda3\condabin\conda.bat" install -y -n pet-eye-gpu -c conda-forge --override-channels cudatoolkit=11.2 cudnn=8.1.0
