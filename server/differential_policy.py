"""Joint classification scores, conditional on five annotated disease classes."""
import numpy as np
from screening_data import DISEASES

CLASSES = list(DISEASES)
NORMAL_CLASSES = CLASSES + ['정상']


def selective_result(findings, info):
    """Display the highest conditional disease score as a reference item."""
    ranked = sorted(findings, key=lambda row: row['score'], reverse=True)
    top = ranked[0]
    if top['disease'] == '정상':
        if not info.get('confirmed_normal_supported'):
            raise ValueError('Normal class requires separately confirmed normal training data.')
        return {'result': 'normal_reference', 'prediction': None, 'top_finding': None,
                'findings': [], 'needs_visit': False, 'reason': 'normal_class_reference',
                'message': '정상 학습 항목에 가까운 사진입니다. 사진만으로 정상이나 질환 부재를 확정할 수는 없습니다.'}
    displayed = {'disease': top['disease'], 'score': top['score'], 'score_points': round(top['score'] * 100, 1)}
    return {'result': 'score_summary', 'prediction': displayed, 'top_finding': displayed,
            'findings': [{'disease': top['disease'], 'score': top['score']}],
            'needs_visit': True, 'reason': 'joint_top_score_reference',
            'message': 'AI 분류 참고 항목입니다. 실제 발병 확률이나 확정 진단이 아닙니다. 수의사에게 눈 상태를 확인해 주세요.'}


def joint_findings(probabilities, info, photo_source):
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    classes = info.get('classes')
    if classes not in (CLASSES, NORMAL_CLASSES) or len(values) != len(classes):
        raise ValueError('Classifier class order does not match disease labels.')
    if classes == NORMAL_CLASSES and not info.get('confirmed_normal_supported'):
        raise ValueError('Normal class has no confirmed training data.')
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1, atol=1e-4):
        raise ValueError('Invalid joint classifier scores.')
    source_evaluated = photo_source in info.get('training_devices', [])
    return [{'disease': disease, 'score': round(float(score), 4),
             'decision': 'classification_reference', 'score_type': 'five_way_classification',
             'validated': False, 'classification_evaluated_for_source': source_evaluated,
             'validation_scope': photo_source,
             'note': '다섯 질환 중 모델 분류 점수이며 정상 여부나 실제 발병 확률을 뜻하지 않습니다.'}
            for disease, score in zip(classes, values)]
