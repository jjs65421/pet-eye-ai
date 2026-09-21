import json
from pathlib import Path
from train import DISEASE_ALIASES, SMOKE_DATA_ROOT, samples_for

sample = next((SMOKE_DATA_ROOT / "train" / "labels").rglob("*.json"))
data = json.loads(sample.read_text(encoding="utf-8"))
print("sample_label=", data["label"]["label_disease_nm"].encode("unicode_escape"))
print("alias=", DISEASE_ALIASES["cataract"].encode("unicode_escape"))
print("image=", (sample.with_name(data["label"]["label_filename"])).is_file())
print("matches=", len(samples_for(SMOKE_DATA_ROOT / "train" / "labels", DISEASE_ALIASES["cataract"])[0]))
