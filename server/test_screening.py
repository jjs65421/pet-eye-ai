"""Regression checks for label selection, leakage and false diagnoses."""
from io import BytesIO
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from image_pipeline import pixels_from_image, pixels_from_path
from screening_data import confirmed_target, partition, DISEASES
from screening_policy import auc, finding, metrics, select_thresholds, summarize, validate_cutoffs
from differential_policy import CLASSES, joint_findings


class ScreeningTests(unittest.TestCase):
    def test_joint_classifier_scores_match_the_correct_disease_labels(self):
        info = {'classes': CLASSES, 'training_devices': ['일반카메라']}
        for index, disease in enumerate(CLASSES):
            scores = np.full(len(CLASSES), .05)
            scores[index] = .8
            result = summarize(joint_findings(scores, info, '스마트폰'))
            self.assertEqual(result['top_finding']['disease'], disease)
            self.assertEqual(result['top_finding']['decision'], 'classification_reference')
            self.assertFalse(result['top_finding']['classification_evaluated_for_source'])

    def test_joint_classifier_rejects_invalid_outputs_and_class_order(self):
        info = {'classes': CLASSES, 'training_devices': []}
        for scores in ([.5, .5], [1, 1, 1, 1], [float('nan'), .2, .3, .5]):
            with self.assertRaises(ValueError):
                joint_findings(scores, info, '스마트폰')
        with self.assertRaises(ValueError):
            joint_findings([.25] * 4, {'classes': CLASSES[::-1]}, '스마트폰')
    def test_missing_labels_are_excluded(self):
        for value in (None, '', 'null', 'unknown'):
            self.assertIsNone(confirmed_target(value))
        self.assertEqual(confirmed_target('무'), 0)
        self.assertEqual(confirmed_target('비성숙'), 1)

    def test_overlap_and_conflicting_labels_removed(self):
        def row(group, split, label, disease='백내장'):
            return {'group': group, 'split': split, 'abnormal': label, 'disease': disease, 'device': '검안경'}
        rows = [row('same', 'training', 1), row('same', 'validation', 1),
                row('same', 'training', 0, '결막염'),
                row('conflict', 'training', 0), row('conflict', 'validation', 1)]
        parts, audit = partition(rows)
        self.assertFalse(parts['train'] or parts['calibration'])
        self.assertEqual(len(parts['test']), 1)
        self.assertEqual(audit['conflicting_image_labels_removed'], 1)
        self.assertEqual(audit['training_images_overlapping_test_removed'], 3)

    def test_unvalidated_score_is_visible_but_not_a_validated_flag(self):
        summary = summarize([finding('백내장', .99, {})])
        self.assertEqual(summary['result'], 'hybrid_screening')
        self.assertEqual(summary['top_finding']['disease'], '백내장')
        self.assertFalse(summary['top_finding']['validated'])
        self.assertEqual(summary['flagged_diseases'], [])
        self.assertTrue(summary['needs_visit'])
        self.assertNotIn('abnormal_probability', summary['top_finding'])

    def test_all_scores_are_sorted_without_hiding_unvalidated_models(self):
        rows = [finding('유루증', .3, {}), finding('백내장', .9, {}), finding('결막염', .6, {})]
        result = summarize(rows)
        self.assertEqual([r['disease'] for r in result['findings']], ['백내장', '결막염', '유루증'])
        self.assertEqual(result['top_finding'], result['findings'][0])
        self.assertEqual(result['score_meaning'], 'independent_model_score_not_disease_probability')

    def test_epiphora_can_be_the_highest_of_all_five_scores(self):
        info = {'thresholds': {'low': .1, 'high': .8}, 'evaluation': {'approved': True}}
        epiphora = finding('유루증', .99, info)
        result = summarize([epiphora, finding('백내장', .2, {}), finding('결막염', .3, {})])
        self.assertEqual(result['top_finding']['disease'], '유루증')
        self.assertEqual(result['epiphora_finding']['decision'], 'possible_abnormality')
        self.assertEqual(result['flagged_diseases'], ['유루증'])
        self.assertEqual(result['ranked_findings'][0]['disease'], '유루증')
        for score, expected in ((.01, 'below_screening_threshold'), (.5, 'inconclusive')):
            result = summarize([finding('유루증', score, info), finding('백내장', .2, {})])
            self.assertEqual(result['epiphora_finding']['decision'], expected)

    def test_independent_cutoffs_allow_multiple_flags(self):
        info = {'thresholds': {'low': .1, 'high': .8}, 'evaluation': {'approved': True}}
        rows = [finding('백내장', .9, info), finding('결막염', .85, info)]
        self.assertEqual(summarize(rows)['flagged_diseases'], ['백내장', '결막염'])
        self.assertEqual(finding('백내장', .5, info)['decision'], 'inconclusive')
        self.assertEqual(finding('백내장', .09, info)['decision'], 'below_screening_threshold')

    def test_low_scores_do_not_establish_health(self):
        info = {'thresholds': {'low': .2, 'high': .8}, 'evaluation': {'approved': True}}
        self.assertEqual(summarize([finding('백내장', .01, info)])['result'], 'hybrid_screening')

    def test_thresholds_require_both_classes_and_test_performance(self):
        labels = np.r_[np.zeros(50), np.ones(50)]
        scores = np.r_[np.linspace(.01, .2, 50), np.linspace(.8, .99, 50)]
        cutoffs = select_thresholds(labels, scores)
        self.assertTrue(validate_cutoffs(labels, scores, cutoffs)['approved'])
        self.assertFalse(validate_cutoffs(labels, 1 - scores, cutoffs)['approved'])
        self.assertIsNone(select_thresholds(np.zeros(100), scores))
        self.assertFalse(validate_cutoffs(np.ones(50), scores[:50], cutoffs)['approved'])

    def test_auc_ties(self):
        self.assertEqual(auc([0, 1, 0, 1], [.5] * 4), .5)
        self.assertEqual(metrics([0, 1], [.5, .5], .5)['fp'], 1)

    def test_training_and_upload_pixels_match(self):
        image = Image.fromarray(np.random.RandomState(42).randint(0, 256, (80, 120, 3), dtype=np.uint8))
        for image_format in ('PNG', 'JPEG'):
            buffer = BytesIO()
            image.save(buffer, format=image_format)
            content = buffer.getvalue()
            with Image.open(BytesIO(content)) as photo:
                upload = pixels_from_image(photo)
            self.assertTrue(np.array_equal(upload, pixels_from_path(BytesIO(content))))
            self.assertEqual(upload.shape, (224, 224, 3))
            self.assertGreater(float(upload.max()), 1)


class ApiTests(unittest.TestCase):
    def test_low_reference_similarity_does_not_block_analysis(self):
        import main
        class Features:
            def numpy(self):
                return np.array([[1., 0.]])
        bundle = {'extractor': lambda *args, **kwargs: Features(),
                  'reference': {'centroid': np.array([0., 1.]), 'threshold': .5}}
        accepted, message, similarity = main.capture_check(np.full((224, 224, 3), 127., dtype=np.float32), bundle)
        self.assertTrue(accepted)
        self.assertIsNotNone(message)
        self.assertEqual(similarity, 0)
        self.assertFalse(main.capture_check(np.zeros((224, 224, 3)), bundle)[0])

    def test_joint_model_replaces_binary_score_ranking(self):
        import main
        bundle = self.bundle()
        class ForbiddenBinaryModel:
            def __call__(self, *args, **kwargs):
                raise AssertionError('Five-disease result must not use independent binary scores')
        for disease in CLASSES:
            bundle['models'][disease] = ForbiddenBinaryModel()
        class JointModel:
            def __call__(self, batch, training=False):
                class Result:
                    def numpy(self):
                        return np.array([[.1, .05, .7, .05, .1]])
                return Result()
        bundle['differential'] = {'model': JointModel(), 'info': {'classes': CLASSES, 'training_devices': ['일반카메라']}}
        bundle['differential_version'] = 'differential-test'
        with patch.object(main, 'load_models', return_value=bundle), \
             patch.object(main, 'capture_check', return_value=(True, None, .8)):
            result = self.run_immediate(main.analyze(self.upload(), '스마트폰'))
        self.assertEqual(result['result'], 'score_summary')
        self.assertEqual(result['prediction']['disease'], '핵경화')
        self.assertEqual(result['prediction']['score'], .7)
        self.assertEqual(result['prediction']['score_points'], 70)
        self.assertEqual(len(result['findings']), 1)
        self.assertNotIn('epiphora_finding', result)
        self.assertTrue(all(set(row) == {'disease', 'score'} for row in result['findings']))
        class LowEpiphoraScore:
            def numpy(self):
                return np.array([[.05]])
        bundle['models']['유루증'] = lambda *args, **kwargs: LowEpiphoraScore()
        with patch.object(main, 'load_models', return_value=bundle), \
             patch.object(main, 'capture_check', return_value=(True, None, .8)):
            result = self.run_immediate(main.analyze(self.upload(), '스마트폰'))
        self.assertEqual(result['prediction']['disease'], '핵경화')
        self.assertEqual(result['prediction']['score_points'], 70)
    @staticmethod
    def upload():
        buffer = BytesIO()
        Image.new('RGB', (60, 60), (127, 127, 127)).save(buffer, format='PNG')
        buffer.seek(0)
        class MemoryUpload:
            content_type = 'image/png'
            async def read(self, size):
                return buffer.read(size)
        return MemoryUpload()

    @staticmethod
    def run_immediate(coroutine):
        # These in-memory coroutines finish without I/O. Windows sandbox blocks
        # asyncio's socketpair initialization, so avoid an unnecessary event loop.
        try:
            coroutine.send(None)
        except StopIteration as complete:
            return complete.value
        raise AssertionError('Unexpected asynchronous I/O in in-memory test')

    @staticmethod
    def bundle():
        class Model:
            def __call__(self, batch, training=False):
                class Result:
                    def numpy(self):
                        return np.array([[.99]])
                return Result()
        info = {'thresholds': {'low': .1, 'high': .8}, 'evaluation': {'approved': True},
                'by_device': {'일반카메라': {'approved': True, 'thresholds': {'low': .1, 'high': .8}}, '스마트폰': {'approved': False}}}
        return {'models': {d: Model() for d in DISEASES},
                'metadata': {d: info for d in DISEASES}, 'version': 'confirmed-test'}

    def test_phone_displays_unvalidated_cataract_score_as_reference(self):
        import main
        with patch.object(main, 'load_models', return_value=self.bundle()), \
             patch.object(main, 'capture_check', return_value=(True, None, .8)):
            result = self.run_immediate(main.analyze(self.upload(), '스마트폰'))
        self.assertEqual(result['result'], 'score_summary')
        self.assertEqual(result['reason'], 'legacy_binary_score_ranking')
        self.assertTrue(result['needs_visit'])
        self.assertEqual(len(result['findings']), 1)
        self.assertIsNotNone(result['top_finding'])
        self.assertNotEqual(result['top_finding']['disease'], '유루증')
        self.assertEqual(result['prediction']['score_points'], 99)
        self.assertNotIn('epiphora_finding', result)
        self.assertNotIn('flagged_diseases', result)
        self.assertTrue(all(set(row) == {'disease', 'score'} for row in result['findings']))

    def test_supported_source_keeps_validated_flags_and_reference_ranking(self):
        import main
        with patch.object(main, 'load_models', return_value=self.bundle()), \
             patch.object(main, 'capture_check', return_value=(True, None, .8)):
            result = self.run_immediate(main.analyze(self.upload(), '일반카메라'))
        self.assertNotIn('flagged_diseases', result)
        self.assertEqual(result['top_finding']['disease'], result['findings'][0]['disease'])

    def test_no_models_is_inconclusive(self):
        import main
        with patch.object(main, 'load_models', return_value=None):
            result = self.run_immediate(main.analyze(self.upload(), '스마트폰'))
        self.assertEqual(result['reason'], 'models_not_ready')
        self.assertFalse(result['needs_visit'])


if __name__ == '__main__':
    unittest.main()
