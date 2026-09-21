"""Index AI Hub label images once so every disease model can reuse the list."""
import json
import os
from collections import Counter
from pathlib import Path


DATA_ROOT = Path(r"D:\153.반려동물 안구질환 데이터\extracted")
MANIFEST_PATH = DATA_ROOT / "labels_manifest.jsonl"
TEMP_PATH = DATA_ROOT / "labels_manifest.pending.jsonl"
NORMAL_VALUES = {"", "무", "정상", "없음", "none", "None", "0"}


def main():
    totals = Counter()
    disease_totals = Counter()
    with TEMP_PATH.open("w", encoding="utf-8") as destination:
        for split in ("training", "validation"):
            label_root = DATA_ROOT / split / "labels"
            for index, json_path in enumerate(label_root.rglob("*.json"), start=1):
                try:
                    with json_path.open("r", encoding="utf-8-sig") as source:
                        annotation = json.load(source)
                    label = annotation["label"]
                    if label.get("label_deleted"):
                        continue
                    image_path = json_path.with_name(label["label_filename"])
                    if not image_path.is_file():
                        continue
                    disease = label.get("label_disease_nm", "")
                    record = {
                        "split": split,
                        "image_path": str(image_path),
                        "disease": disease,
                        "abnormal": int(str(label.get("label_disease_lv_1", "")).strip() not in NORMAL_VALUES),
                        "species": annotation.get("images", {}).get("meta", {}).get("breed", ""),
                    }
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                    totals[split] += 1
                    disease_totals[(split, disease, record["abnormal"])] += 1
                except (OSError, UnicodeDecodeError, ValueError, KeyError):
                    continue
                if index % 50_000 == 0:
                    print(f"indexed_{split}={index}", flush=True)
    os.replace(TEMP_PATH, MANIFEST_PATH)
    print(f"manifest={MANIFEST_PATH}", flush=True)
    for split in ("training", "validation"):
        print(f"samples_{split}={totals[split]}", flush=True)
    for key, count in sorted(disease_totals.items()):
        print(f"count={key[0]}|{key[1]}|{key[2]}|{count}", flush=True)


if __name__ == "__main__":
    main()
