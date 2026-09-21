"""Create a small balanced cataract data set directly from AI Hub label ZIPs."""
import json
import shutil
import zipfile
from pathlib import Path


ARCHIVES = {
    "train": [
        Path(r"D:\153.반려동물 안구질환 데이터\01.데이터\1.Training\라벨링데이터\TL1.zip"),
        Path(r"D:\153.반려동물 안구질환 데이터\01.데이터\1.Training\라벨링데이터\TL2.zip"),
    ],
    "validation": [Path(r"D:\153.반려동물 안구질환 데이터\01.데이터\2.Validation\라벨링데이터\VL.zip")],
}
OUTPUT = Path(r"D:\153.반려동물 안구질환 데이터\smoke-cataract")


def abnormal(level):
    return str(level).strip() not in {"", "무", "정상", "없음", "none", "None", "0"}


def prepare(split, per_class):
    counts = {0: 0, 1: 0}
    for archive_path in ARCHIVES[split]:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".json") or counts[0] >= per_class and counts[1] >= per_class:
                    continue
                try:
                    data = json.loads(archive.read(info).decode("utf-8-sig"))
                    label = data["label"]
                    if label.get("label_deleted") or label.get("label_disease_nm") != "백내장":
                        continue
                    target = int(abnormal(label.get("label_disease_lv_1")))
                    if counts[target] >= per_class:
                        continue
                    image_member = info.filename.rsplit("/", 1)[0] + "/" + label["label_filename"]
                    image_data = archive.read(image_member)
                except (KeyError, OSError, UnicodeDecodeError, ValueError):
                    continue
                folder = OUTPUT / split / "labels" / str(target)
                folder.mkdir(parents=True, exist_ok=True)
                stem = f"{archive_path.stem}_{counts[target]:05d}"
                image_name = f"{stem}.jpg"
                data["label"]["label_filename"] = image_name
                (folder / image_name).write_bytes(image_data)
                (folder / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                counts[target] += 1
    print(f"{split}: normal={counts[0]}, abnormal={counts[1]}", flush=True)
    if min(counts.values()) < per_class:
        raise RuntimeError(f"Not enough {split} samples: {counts}")


if __name__ == "__main__":
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    prepare("train", per_class=1000)
    prepare("validation", per_class=250)
