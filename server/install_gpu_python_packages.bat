@echo off
call "C:\Users\jungc\miniconda3\condabin\conda.bat" run -n pet-eye-gpu python -m pip install --upgrade pip
call "C:\Users\jungc\miniconda3\condabin\conda.bat" run -n pet-eye-gpu python -m pip install tensorflow==2.7.0 protobuf==3.20.3 fastapi uvicorn[standard] python-multipart pillow
