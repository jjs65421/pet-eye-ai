"""Audit metadata and crop geometry without changing images or medical labels."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from screening_data import read_records, partition, SERVER_ROOT

splits, _ = partition(read_records())
rng = np.random.RandomState(19)
cache = {}
report = {}
for split, rows in splits.items():
    report[split] = {}
    for disease in ('유루증', '안검염'):
        selected = [r for r in rows if r['disease'] == disease and r['abnormal'] == 1 and r['device'] == '일반카메라']
        sample = [selected[i] for i in rng.permutation(len(selected))[:100]]
        months, labels, species, crop_area, ages = [], [], [], [], []
        for row in sample:
            path = Path(row['image_path'])
            if path.parent not in cache:
                mapping = {}
                for source in path.parent.glob('*.json'):
                    annotation = json.loads(source.read_text(encoding='utf-8-sig'))
                    mapping[annotation.get('label', {}).get('label_filename')] = annotation
                cache[path.parent] = mapping
            annotation = cache[path.parent].get(path.name)
            if annotation is None:
                raise RuntimeError('Missing original annotation')
            meta, label = annotation['images']['meta'], annotation['label']
            assert label['label_disease_nm'] == disease
            assert label['label_disease_lv_3'] == row['confirmed_label']
            labels.append((str(label.get('label_disease_lv_1')), str(label.get('label_disease_lv_3'))))
            species.append('cat' if '/고양이/' in label['label_path'] else 'dog' if '/개/' in label['label_path'] else 'unknown')
            months.append(str(meta.get('date_time'))[:7])
            ages.append(float(meta['age']))
            width, height = meta['width_height']
            box = label['label_bbox']
            crop_area.append(float(box[2] * box[3] / (width * height)))
        report[split][disease] = {'population': len(selected), 'sample': len(sample),
            'species': dict(Counter(species)), 'capture_months': dict(Counter(months)),
            'first_final_disagreements': sum(a != b for a, b in labels),
            'age_quartiles': np.quantile(ages, [.25, .5, .75]).tolist(),
            'bbox_area_fraction_quartiles': np.quantile(crop_area, [.25, .5, .75]).tolist()}
output = SERVER_ROOT / 'training_data' / 'capture-condition-audit.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
