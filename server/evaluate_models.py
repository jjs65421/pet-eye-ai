"""Recompute recorded holdout metrics without changing deployed thresholds."""
import json
from pathlib import Path
import numpy as np
from screening_policy import validate_cutoffs
from screening_data import DISEASES


def main():
    root = Path(__file__).parent / "models"
    pointer = json.loads((root / "active_models.json").read_text(encoding="utf-8"))
    active = root / pointer["version"]
    for disease in DISEASES:
        info_path = active / f'{disease}.json'
        info = json.loads(info_path.read_text(encoding="utf-8"))
        predictions = np.load(active / f"{info['disease']}-test-predictions.npz")
        evaluation = validate_cutoffs(predictions["labels"], predictions["scores"], info["thresholds"])
        assert evaluation == info["aggregate_evaluation"], f"Stored metrics differ: {info['disease']}"
        for device, expected in info['by_device'].items():
            if 'thresholds' not in expected:
                continue
            mask = predictions['devices'] == device
            actual = validate_cutoffs(predictions['labels'][mask], predictions['scores'][mask], expected['thresholds'])
            assert actual == {k: v for k, v in expected.items() if k != 'thresholds'}, f"Device metrics differ: {info['disease']}/{device}"
        print(info["disease"], {'deployment_eligible': info['evaluation']['approved'],
              'eligible_sources': [device for device, result in info['by_device'].items() if result['approved']],
              'aggregate_roc_auc': evaluation.get('roc_auc')}, flush=True)


if __name__ == "__main__":
    main()
