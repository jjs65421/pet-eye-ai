# 노트북 이전 안내

## 1. 복사할 폴더

`pet-eye-ai` 폴더 전체를 외장 SSD 또는 OneDrive로 새 노트북에 복사합니다.

반드시 포함할 항목:

- `App.js`, `app.json`, `package.json`, `package-lock.json`
- `server/main.py`, `server/run_server.bat`, `server/requirements.txt`
- `server/models/` 폴더 전체 (5개 AI 모델과 촬영 적합성 기준 파일)

`node_modules`, `dist`, `*.log`, `server/.venv`는 복사하지 않아도 됩니다.

AI Hub 원본 데이터(D: 드라이브의 약 200GB)는 **재학습할 때만** 필요합니다. 시연용 AI 분석에는 옮길 필요가 없습니다.

## 2. 새 노트북에서 설치

1. Node.js LTS와 Expo Go(휴대폰)를 설치합니다.
2. 프로젝트 폴더에서 `npm install`을 실행합니다.
3. Python 3.8 환경을 만들고 `pip install -r server/requirements.txt`를 실행합니다.
   GPU를 사용하려면 현재 노트북과 같은 CUDA/TensorFlow GPU 환경을 별도로 설치해야 합니다. GPU가 없어도 서버는 CPU로 실행할 수 있지만 분석이 느려집니다.

## 3. 실행

1. `server/run_server.bat`를 실행하여 AI 서버를 시작합니다.
2. 다른 터미널에서 `npx expo start --host lan`을 실행합니다.
3. Expo Go에서 QR 코드를 스캔합니다.

## 4. 꼭 바꿀 값

새 노트북은 Wi-Fi IP가 달라집니다. `App.js`의 `AI_SERVER_URL`을 새 노트북 IPv4 주소로 바꾸세요.

예: `const AI_SERVER_URL = 'http://192.168.0.15:8000';`

시연 전에 휴대폰 브라우저에서 `http://새_IP:8000/health`를 열어 `ready`가 보이는지 확인합니다.
