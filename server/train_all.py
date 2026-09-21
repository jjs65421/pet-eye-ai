"""Run full-data training sequentially for every disease after indexing completes."""
import subprocess
import sys
import time
from pathlib import Path


DATA_ROOT = Path(r"D:\153.반려동물 안구질환 데이터\extracted")
MANIFEST = DATA_ROOT / "labels_manifest.jsonl"
DISEASES = ["cataract", "epiphora", "sclerosis", "blepharitis", "conjunctivitis", "corneal_ulcer"]
TRAIN_SCRIPT = Path(__file__).parent / "train.py"


def main():
    while not MANIFEST.is_file():
        print("waiting_for_manifest", flush=True)
        time.sleep(30)
    for disease in DISEASES:
        print(f"starting={disease}", flush=True)
        subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--disease", disease, "--epochs", "5"], check=True)
        print(f"completed={disease}", flush=True)
    print("all_models_completed", flush=True)


if __name__ == "__main__":
    main()
