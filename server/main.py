"""Local FastAPI inference server for the Pet Eye Check school demonstration.

Images are decoded in memory only.  They are not written to disk by this API.
"""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


MODEL_DIR = Path(__file__).parent / "models"
EYE_REFERENCE_PATH = MODEL_DIR / "eye_photo_reference.npz"
# The corneal-ulcer model did not pass validation, so it is deliberately not
# exposed by the demo API.  Its artefact is kept only for troubleshooting.
EXCLUDED_MODEL_STEMS = {"각막궤양"}
models = {}
metadata = {}
eye_feature_extractor = None
eye_reference = None
app = FastAPI(title="Pet Eye Check AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # School-demo LAN only; restrict this before public deployment.
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def load_models():
    global eye_feature_extractor, eye_reference
    if models:
        return
    for model_path in MODEL_DIR.glob("*.keras"):
        key = model_path.stem
        if key in EXCLUDED_MODEL_STEMS:
            continue
        models[key] = tf.keras.models.load_model(model_path)
        info_path = model_path.with_suffix(".json")
        metadata[key] = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"disease": key}
    reference = np.load(EYE_REFERENCE_PATH)
    eye_reference = {"centroid": reference["centroid"], "threshold": float(reference["threshold"])}
    reference_model = models["백내장"]
    eye_feature_extractor = tf.keras.Model(reference_model.input, reference_model.get_layer("dropout").output)


def capture_check(photo):
    pixels = np.asarray(photo, dtype=np.float32)
    brightness = float(pixels.mean())
    if brightness < 25 or brightness > 245:
        return False, "사진이 너무 어둡거나 밝아요. 밝은 곳에서 반려동물의 눈을 다시 촬영해 주세요.", None
    batch = np.expand_dims(pixels, axis=0)
    feature = eye_feature_extractor.predict(batch, verbose=0)[0]
    feature /= max(float(np.linalg.norm(feature)), 1e-8)
    similarity = float(feature @ eye_reference["centroid"])
    if similarity < eye_reference["threshold"]:
        return False, "반려동물의 눈이 선명하게 보이는 사진인지 확인해 주세요. 눈 주변이 화면 중앙에 오도록 다시 촬영해 주세요.", similarity
    return True, None, similarity


@app.get("/health")
def health():
    load_models()
    return {"status": "ready" if models else "waiting_for_trained_model", "models": [info["disease"] for info in metadata.values()]}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    load_models()
    if not models:
        raise HTTPException(503, "학습된 모델이 아직 없습니다.")
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "JPG, PNG 또는 WEBP 사진만 분석할 수 있습니다.")
    try:
        photo = Image.open(BytesIO(await image.read())).convert("RGB").resize((224, 224))
    except Exception as error:
        raise HTTPException(400, "사진을 읽을 수 없습니다.") from error
    accepted, retry_message, similarity = capture_check(photo)
    if not accepted:
        return {
            "result": "retake_required",
            "capture_check": {"accepted": False, "message": retry_message, "eye_photo_similarity": round(similarity, 4) if similarity is not None else None},
            "image_saved": False,
        }
    batch = np.expand_dims(np.asarray(photo, dtype=np.float32), axis=0)
    findings = []
    for key, model in models.items():
        probability = float(model.predict(batch, verbose=0)[0][0])
        findings.append({"disease": metadata[key]["disease"], "abnormal_probability": round(probability, 4)})
    findings.sort(key=lambda item: item["abnormal_probability"], reverse=True)
    return {
        "result": "screening_only",
        "top_finding": findings[0],
        "findings": findings,
        "capture_check": {"accepted": True, "eye_photo_similarity": round(similarity, 4)},
        "image_saved": False,
        "notice": "이 결과는 질환 진단이 아닌 이상 가능성 선별 결과입니다.",
    }
