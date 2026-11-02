"""Dataset-validated screening. Images are processed in memory and never saved."""
from io import BytesIO
import json
from pathlib import Path
from threading import RLock

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from image_pipeline import PREPROCESSING, pixels_from_image
from model_loading import load_screening_model
from screening_data import DISEASES
from screening_policy import finding, summarize
from differential_policy import CLASSES, NORMAL_CLASSES, joint_findings, selective_result

MODEL_DIR = Path(__file__).parent / "models"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_bundle = None
_lock = RLock()
app = FastAPI(title="Pet Eye Check AI", version="0.6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"], allow_headers=["*"])


def load_models():
    global _bundle
    pointer = MODEL_DIR / "active_models.json"
    if not pointer.is_file():
        return None
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    differential_pointer = MODEL_DIR / 'active_differential.json'
    differential_version = json.loads(differential_pointer.read_text(encoding='utf-8'))['version'] if differential_pointer.is_file() else None
    if differential_version and (Path(differential_version).name != differential_version or not differential_version.startswith('differential-')):
        raise RuntimeError('Invalid differential model version')
    # Only a direct child under models can be selected by the local run pointer.
    if Path(version).name != version or not version.startswith("confirmed-"):
        raise RuntimeError("Invalid model version")
    with _lock:
        if _bundle is not None and _bundle["version"] == version and _bundle.get('differential_version') == differential_version:
            return _bundle
        if _bundle is None:
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # A caller may already have initialized the TF runtime.
        folder = MODEL_DIR / version
        models, metadata = {}, {}
        for disease in DISEASES:
            info = json.loads((folder / f"{disease}.json").read_text(encoding="utf-8"))
            if info.get("schema") != "screening-v2" or info.get("preprocessing") != PREPROCESSING:
                raise RuntimeError(f"Incompatible preprocessing/metadata: {disease}")
            if info.get("positive_class") != 1 or info.get("label_field") != "label_disease_lv_3":
                raise RuntimeError(f"Incompatible label mapping: {disease}")
            models[disease] = load_screening_model(folder / f"{disease}.h5")
            metadata[disease] = info
        with np.load(folder / "eye_photo_reference.npz") as reference:
            quality = {"centroid": reference["centroid"].copy(), "threshold": float(reference["threshold"])}
        _bundle = {"version": version, "models": models, "metadata": metadata, "reference": quality,
                   "extractor": load_screening_model(folder / "feature_extractor.h5")}
        if differential_version:
            differential_folder = MODEL_DIR / differential_version
            info = json.loads((differential_folder / 'classifier.json').read_text(encoding='utf-8'))
            if info.get('schema') != 'differential-v1' or info.get('preprocessing') != PREPROCESSING or info.get('classes') not in (CLASSES, NORMAL_CLASSES):
                raise RuntimeError('Incompatible joint classifier metadata')
            if info['classes'] == NORMAL_CLASSES and not info.get('confirmed_normal_supported'):
                raise RuntimeError('Normal classifier lacks confirmed normal data')
            if not info.get('accepted_for_reference_display') and not info.get('experimental_forced_selection'):
                raise RuntimeError('Joint classifier did not pass development evaluation')
            _bundle['differential'] = {'model': load_screening_model(differential_folder / 'classifier.h5'), 'info': info}
        _bundle['differential_version'] = differential_version
        return _bundle


def capture_check(pixels, bundle):
    brightness = float(pixels.mean())
    if brightness < 25 or brightness > 245:
        return False, "사진이 너무 어둡거나 밝아요. 밝은 곳에서 눈을 다시 촬영해 주세요.", None
    feature = bundle["extractor"](np.expand_dims(pixels, 0), training=False).numpy()[0]
    feature /= max(float(np.linalg.norm(feature)), 1e-8)
    similarity = float(feature @ bundle["reference"]["centroid"])
    if similarity < bundle["reference"]["threshold"]:
        # An unvalidated centroid check cannot reliably reject a real eye photo.
        return True, "학습 사진과 촬영 방식이 달라 분석 점수는 참고용으로만 확인해 주세요.", similarity
    return True, None, similarity


@app.get("/health")
def health():
    bundle = load_models()
    if bundle is None:
        return {"status": "waiting_for_confirmed_models", "models": [], "smartphone_validated_models": []}
    return {"status": "ready", "model_version": bundle["version"], "models": list(bundle["models"]),
            'classification_model_version': bundle.get('differential_version'),
            'disease_method': 'joint_five_classification' if bundle.get('differential') else 'legacy_binary_score_ranking',
            'normal_class_available': bool(bundle.get('differential', {}).get('info', {}).get('confirmed_normal_supported')),
            'decision_policy': 'always_show_top_score_v1',
            "validated_models": [d for d, i in bundle["metadata"].items() if i["evaluation"]["approved"]],
            "smartphone_validated_models": [d for d, i in bundle["metadata"].items() if i["smartphone_validated"]]}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), photo_source: str = Form("스마트폰")):
    """photo_source: 스마트폰 (default), 일반카메라, or 검안경.

    Only select a different source when the image was actually captured that way.
    All model scores are shown as references. Source-specific validation only
    controls decision labels; it does not hide a model's score.
    """
    if photo_source not in {"스마트폰", "일반카메라", "검안경"}:
        raise HTTPException(400, "촬영 장비는 스마트폰, 일반카메라 또는 검안경이어야 합니다.")
    bundle = load_models()
    if bundle is None:
        return {"result": "inconclusive", "reason": "models_not_ready", "flagged_diseases": [],
                "needs_visit": False, "top_finding": None, "findings": [], "image_saved": False,
                "message": "확정 라벨로 모델을 학습·검증 중입니다. 현재는 질환을 판단할 수 없어요."}
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "JPG, PNG 또는 WEBP 사진만 분석할 수 있습니다.")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "15MB 이하 사진을 사용해 주세요.")
    try:
        with Image.open(BytesIO(content)) as photo:
            pixels = pixels_from_image(photo)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "사진을 읽을 수 없습니다.") from error
    accepted, message, similarity = capture_check(pixels, bundle)
    capture = {"accepted": accepted, "message": message,
               "eye_photo_similarity": round(similarity, 4) if similarity is not None else None,
               "note": "Image similarity is an input check, not proof of an eye or diagnosis."}
    if not accepted:
        return {'result': 'inconclusive', 'reason': 'poor_photo_quality', 'prediction': None,
                'top_finding': None, 'findings': [], 'needs_visit': False,
                'capture_check': capture, 'image_saved': False, 'model_version': bundle['version'],
                'message': message + ' 이 사진으로는 판단하기 어려워요. 정상 판정이 아닙니다.'}
    batch = np.expand_dims(pixels, 0)
    findings = []
    diseases = [] if bundle.get('differential') else DISEASES
    for disease in diseases:
        score = float(bundle["models"][disease](batch, training=False).numpy().reshape(-1)[0])
        if not np.isfinite(score):
            raise HTTPException(503, "모델 출력이 올바르지 않습니다.")
        info = bundle["metadata"][disease]
        device_evaluation = info.get("by_device", {}).get(photo_source, {"approved": False})
        effective_info = dict(info)
        effective_info['thresholds'] = device_evaluation.get('thresholds')
        effective_info["evaluation"] = {"approved": bool(info["evaluation"]["approved"] and device_evaluation["approved"])}
        row = finding(disease, score, effective_info)
        row.pop('thresholds', None)  # Calibration cutoffs are internal, not photo results.
        row["validation_scope"] = photo_source
        findings.append(row)
    if bundle.get('differential'):
        classifier = bundle['differential']
        values = classifier['model'](batch, training=False).numpy()[0]
        try:
            findings.extend(joint_findings(values, classifier['info'], photo_source))
        except ValueError as error:
            raise HTTPException(503, '질환 분류 모델 출력을 확인할 수 없습니다.') from error
    summary = summarize(findings)
    is_joint = bool(bundle.get('differential'))
    # All five scores now come from the same softmax classifier.
    summary.pop('epiphora_finding', None)
    summary.pop('ranked_findings', None)
    summary.pop('flagged_diseases', None)
    top = summary['top_finding']
    displayed = {'disease': top['disease'], 'score': top['score'], 'score_points': round(top['score'] * 100, 1)}
    summary['result'] = 'score_summary'
    summary['top_finding'] = displayed
    summary['prediction'] = displayed
    summary['findings'] = [{'disease': top['disease'], 'score': top['score']}]
    summary['score_meaning'] = 'model_score_not_disease_probability'
    summary['ranking_notice'] = '하나의 모델이 다섯 질환을 비교한 점수입니다. 정상 사진에서도 항목 하나가 선택되며 실제 발병 확률은 아닙니다.' if is_joint else '모델별 점수 척도가 달라 최고 점수가 실제 질환 가능성 순위를 뜻하지는 않습니다.'
    if is_joint:
        summary.update(selective_result(findings, bundle['differential']['info']))
    else:
        summary.update({'result': 'inconclusive', 'prediction': None, 'top_finding': None,
                        'findings': [], 'needs_visit': False, 'reason': 'joint_model_unavailable',
                        'message': '질환을 비교하는 모델이 준비되지 않아 판단하기 어려워요.'})
    summary['ranking_notice'] = '다섯 질환 중 점수가 가장 높은 항목을 참고 결과로 표시합니다. 점수는 실제 발병 확률이나 진단 정확도가 아닙니다.'
    return {**summary, "capture_check": capture, "image_saved": False,
            "model_version": bundle["version"], "photo_source": photo_source,
            'classification_model_version': bundle.get('differential_version'),
            'experimental_model': bool(bundle.get('differential', {}).get('info', {}).get('experimental_forced_selection')),
            'normal_class_available': bool(bundle.get('differential', {}).get('info', {}).get('confirmed_normal_supported')),
            'decision_policy': {'version': 'always_show_top_score_v1', 'calibrated': False},
            "notice": "점수는 질환 확률이나 신뢰도가 아닙니다. 검증되지 않은 항목도 참고용으로 표시하며 정상 판정 또는 확정 진단을 제공하지 않습니다."}
