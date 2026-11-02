"""Confirmed labels and image-level splits; patient identifiers are unavailable."""
import hashlib
import json
from pathlib import Path

SERVER_ROOT = Path(__file__).parent
DEFAULT_DATA_ROOT = Path(r"D:\153.반려동물 안구질환 데이터\extracted")
MANIFEST_PATH = SERVER_ROOT / "training_data" / "confirmed_manifest.jsonl"
DISEASE_ALIASES = {
    "cataract": "백내장", "epiphora": "유루증", "sclerosis": "핵경화",
    "blepharitis": "안검염", "conjunctivitis": "결막염",
}
DISEASES = list(DISEASE_ALIASES.values())
NORMAL = {"무", "정상", "없음", "0"}
POSITIVE = {"유", "상", "하", "초기", "비성숙", "성숙", "1"}
SCHEMA = "confirmed-lv3-v1"


def confirmed_target(value):
    value = str(value).strip() if value is not None else ""
    if value in NORMAL:
        return 0
    if value in POSITIVE:
        return 1
    return None


def annotation_record(json_path, split):
    annotation = json.loads(json_path.read_text(encoding="utf-8-sig"))
    label = annotation["label"]
    meta = annotation.get("images", {}).get("meta", {})
    target = confirmed_target(label.get("label_disease_lv_3"))
    image = json_path.with_name(label["label_filename"])
    if label.get("label_deleted") or target is None or not image.is_file():
        return None
    device = str(meta.get("device", "unknown"))
    if "초음파" in device:
        return None
    return {
        "schema": SCHEMA, "split": split, "image_path": str(image),
        "group": str(meta.get("file_name") or image.name).casefold(),
        "disease": label.get("label_disease_nm"), "abnormal": target,
        "confirmed_label": label.get("label_disease_lv_3"), "device": device,
        "breed": str(meta.get("breed", "unknown")),
    }


def read_records(path=MANIFEST_PATH):
    with Path(path).open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    if not records or any(row.get("schema") != SCHEMA for row in records):
        raise ValueError("Rebuild the confirmed-lv3 manifest before training.")
    return records


def partition(records):
    test_groups = {row["group"] for row in records if row["split"] == "validation"}
    # Official validation has no smartphone positives. Reserve existing training
    # phone images BEFORE fitting; group assignment also covers other diseases.
    phone_groups = {r['group'] for r in records if r['split'] == 'training' and r['device'] == '스마트폰'}
    supplementary_test = {g for g in phone_groups
                          if 15 <= int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 100 < 30}
    result = {"train": [], "calibration": [], "test": []}
    seen, conflicts = {}, set()
    for row in records:
        key = (row["group"], row["disease"])
        if key in seen and seen[key]["abnormal"] != row["abnormal"]:
            conflicts.add(key)
        elif key not in seen or row["split"] == "validation":
            seen[key] = row
    overlap_removed = len({(r['group'], r['disease']) for r in records
                           if r['split'] == 'training' and r['group'] in test_groups})
    for key, row in seen.items():
        if key in conflicts:
            continue
        if row["split"] == "validation" or row['group'] in supplementary_test:
            name = "test"
        elif row["group"] in test_groups:
            continue
        else:
            bucket = int(hashlib.sha256(row["group"].encode()).hexdigest()[:8], 16) % 100
            name = "calibration" if bucket < 15 else "train"
        result[name].append(row)
    return result, {"conflicting_image_labels_removed": len(conflicts),
                    "duplicate_annotation_rows_removed": len(records) - len(seen),
                    "training_images_overlapping_test_removed": overlap_removed,
                    "supplementary_phone_test_groups": len(supplementary_test),
                    "split_unit": "original image filename; patient identity unavailable"}
