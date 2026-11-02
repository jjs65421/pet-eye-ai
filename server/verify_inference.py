"""Check exported models against cached holdout scores and exercise real API logic."""
import json
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import main
from image_pipeline import pixels_from_path
from screening_data import DISEASES, SERVER_ROOT, read_records, partition
from differential_policy import CLASSES
from test_screening import ApiTests


def run():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    bundle = main.load_models()
    assert bundle is not None
    parts, _ = partition(read_records())
    output = SERVER_ROOT / 'models' / bundle['version']
    agreements = {}
    for disease in DISEASES:
        info = bundle['metadata'][disease]
        rows = [r for r in parts['test'] if r['disease'] == disease and r['device'] in info['training_devices']]
        expected = np.load(output / f'{disease}-test-predictions.npz')['scores']
        positions = [0, len(rows) // 2, len(rows) - 1]
        batch = np.stack([pixels_from_path(rows[i]['image_path']) for i in positions])
        actual = bundle['models'][disease](batch, training=False).numpy().reshape(-1)
        error = float(np.max(np.abs(actual - expected[positions])))
        assert error < .0001, (disease, error)
        agreements[disease] = error
    if bundle.get('differential'):
        classifier = bundle['differential']
        differential_output = SERVER_ROOT / 'models' / bundle['differential_version']
        rows = [r for r in parts['test'] if r['disease'] in CLASSES and r['abnormal'] == 1
                and r['device'] in classifier['info']['training_devices']]
        expected = np.load(differential_output / 'test-predictions.npz')['scores']
        positions = [next(i for i, r in enumerate(rows) if r['disease'] == disease) for disease in CLASSES]
        images = np.stack([pixels_from_path(rows[i]['image_path']) for i in positions])
        actual = classifier['model'](images, training=False).numpy()
        error = float(np.max(np.abs(actual - expected[positions])))
        assert error < .0001, error
        agreements['joint_four_disease_classifier'] = error
    responses = []
    epiphora_rows = [r for r in parts['test'] if r['disease'] == '유루증']
    epiphora_scores = np.load(output / '유루증-test-predictions.npz')['scores']
    high = bundle['metadata']['유루증']['by_device']['스마트폰']['thresholds']['high']
    for label, case in ((0, 'first_negative'), (1, 'first_positive'), (1, 'positive_above_cutoff')):
        row = next(r for i, r in enumerate(epiphora_rows) if r['device'] == '스마트폰' and r['abnormal'] == label
                   and (case != 'positive_above_cutoff' or epiphora_scores[i] >= high))
        with open(row['image_path'], 'rb') as source:
            content = source.read()
        class Upload:
            content_type = 'image/jpeg'
            async def read(self, size):
                return content[:size]
        result = ApiTests.run_immediate(main.analyze(Upload(), '스마트폰'))
        if result['result'] != 'retake_required':
            assert all(set(row) == {'disease', 'score'} for row in result['findings'])
            assert result['result'] == 'score_summary'
            assert len(result['findings']) == 1
            assert result['top_finding']['disease'] == result['findings'][0]['disease']
            assert result['prediction']['score_points'] == round(result['top_finding']['score'] * 100, 1)
            assert 'epiphora_finding' not in result
        responses.append({'case': case, 'confirmed_epiphora_label': label, 'response': result})
    report = {'model_version': bundle['version'], 'export_score_max_error': agreements,
              'sample_api_responses': responses, 'health': main.health()}
    (output / 'inference_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if bundle.get('differential'):
        (SERVER_ROOT / 'models' / bundle['differential_version'] / 'inference_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
