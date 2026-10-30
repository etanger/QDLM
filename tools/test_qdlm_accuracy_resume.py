"""CPU tests for recovery, cache integrity, and completion validation."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qdlm_accuracy_resume import (
    RUNTIME_FIELDS, SOURCE_PATHS, load_records, replay_check_keys, request_key,
    require_idle_gpus, stream_predictions, validate_completion, validate_requests,
    validate_settings, validate_sources, verify_cached_predictions, write_json,
)


def requests(n=6):
    return [SimpleNamespace(task_name="mmlu_test", doc_id=i, idx=0,
                            args=(f"Question {i}", {"until": ["STOP"]})) for i in range(n)]


def prediction(request):
    return dict(case_id=request_key(request), task=request.task_name, doc_id=request.doc_id,
                answer=str(request.doc_id), generated_token_ids=[request.doc_id, 7])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.requests = requests()
        self.rows = [prediction(r) for r in self.requests]

    def tearDown(self):
        self.temp.cleanup()

    def save_rows(self, rows=None, tail=b""):
        path = self.root / "outputs.jsonl"
        path.write_bytes(b"".join((json.dumps(row) + "\n").encode()
                                  for row in (self.rows if rows is None else rows)) + tail)
        return path

    def test_request_key_is_compatible_with_original_evaluator(self):
        r = self.requests[0]
        old = hashlib.sha256(json.dumps([r.task_name, r.doc_id, r.idx, r.args],
                                       sort_keys=True).encode()).hexdigest()
        self.assertEqual(request_key(r), old)

    def test_only_missing_cases_generated_and_harness_order_preserved(self):
        cached = {self.rows[i]["case_id"]: self.rows[i] for i in (0, 2, 5)}
        calls, advanced = [], []
        def predict(r):
            calls.append(r.doc_id)
            return prediction(r)
        handle = io.StringIO()
        answers = stream_predictions(self.requests, cached, predict, handle,
                                     lambda: advanced.append(1))
        self.assertEqual(calls, [1, 3, 4])
        self.assertEqual(answers, [str(i) for i in range(6)])
        saved = [json.loads(line) for line in handle.getvalue().splitlines()]
        self.assertEqual(len({r['case_id'] for r in saved}), 6)
        self.assertEqual(len(advanced), 3)

    def test_second_interruption_retains_cached_and_new_predictions(self):
        cached = {self.rows[0]["case_id"]: self.rows[0]}
        def predict(r):
            if r.doc_id == 3:
                raise RuntimeError("interrupted")
            return prediction(r)
        path = self.root / 'outputs.jsonl'
        with path.open('w') as handle:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                stream_predictions(self.requests, cached, predict, handle, lambda: None)
        recovered, _ = load_records(path, 2)
        self.assertEqual({r['doc_id'] for r in recovered.values()}, {0, 1, 2})

    def test_replay_checks_first_middle_last_and_rejects_changed_tokens(self):
        cached = {row['case_id']: row for row in self.rows}
        keys = [request_key(r) for r in self.requests]
        self.assertEqual(replay_check_keys(keys, cached), [keys[0], keys[3], keys[5]])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify_cached_predictions(self.requests, cached, prediction),
                             [keys[0], keys[3], keys[5]])
            def changed(r):
                return dict(prediction(r), generated_token_ids=[999, 7])
            with self.assertRaisesRegex(ValueError, 'refusing to mix'):
                verify_cached_predictions(self.requests, cached, changed)

    def test_duplicate_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_records(self.save_rows([self.rows[0], self.rows[0]]), 2)

    def test_torn_final_line_ignored_without_modifying_source(self):
        path = self.save_rows(tail=b'{"case_id":')
        before = path.read_bytes()
        loaded, ignored = load_records(path, 2, allow_torn_final=True)
        self.assertEqual(len(loaded), 6)
        self.assertTrue(ignored)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            load_records(path, 2)

    def test_corrupt_complete_line_and_wrong_token_count_rejected(self):
        with self.assertRaises(ValueError):
            load_records(self.save_rows(tail=b'broken\n'), 2, allow_torn_final=True)
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            load_records(self.save_rows([dict(self.rows[0], generated_token_ids=[7])]), 2)

    def test_prompt_change_and_total_count_change_rejected(self):
        cached = {self.rows[0]['case_id']: self.rows[0]}
        changed = requests()
        changed[0].args = ('Different prompt', {'until': ['STOP']})
        with self.assertRaisesRegex(ValueError, 'prompt'):
            validate_requests(changed, cached)
        with self.assertRaisesRegex(ValueError, 'count'):
            validate_requests(self.requests, cached, {'request_count': 7})

    def test_setting_and_runtime_mismatch_rejected(self):
        args = dict(model='same-model', variant='quantized', tasks='mmlu_generative',
                    num_fewshot=5, limit=80, steps=128, gen_length=64, block_length=64)
        old = dict(args=args, config={'stage': 'w23'}, **{k: 'same' for k in RUNTIME_FIELDS})
        current = json.loads(json.dumps(old))
        validate_settings(old, current)
        current['args']['steps'] = 64
        with self.assertRaisesRegex(ValueError, 'steps'):
            validate_settings(old, current)
        current = json.loads(json.dumps(old))
        current['torch'] = 'different'
        with self.assertRaisesRegex(ValueError, 'torch'):
            validate_settings(old, current)

    def test_source_hash_mismatch_and_dirty_legacy_source_rejected(self):
        hashes = {name: 'same' for name in SOURCE_PATHS}
        validate_sources({'source_sha256': hashes}, {'source_sha256': hashes}, self.root)
        altered = dict(hashes, **{SOURCE_PATHS[0]: 'changed'})
        with self.assertRaisesRegex(ValueError, 'source changed'):
            validate_sources({'source_sha256': hashes}, {'source_sha256': altered}, self.root)
        with self.assertRaisesRegex(ValueError, 'locally|modified'):
            validate_sources({'source_revision': 'a' * 40, 'local_changes': ' M ' + SOURCE_PATHS[0]},
                             {'source_sha256': hashes}, self.root)

    def make_complete(self):
        self.save_rows()
        write_json(self.root / 'metadata.json', dict(status='complete', request_count=6,
                                                    args={'gen_length': 2}))
        write_json(self.root / 'accuracy.json', {'results': {'mmlu_generative': {'score': 0.5}}})
        write_json(self.root / 'complete.json', dict(status='complete', completed_requests=6,
            sha256={n: hashlib.sha256((self.root / n).read_bytes()).hexdigest()
                    for n in ('outputs.jsonl', 'accuracy.json')}))

    def test_complete_run_accepted_and_incomplete_marker_rejected(self):
        self.make_complete()
        self.assertEqual(validate_completion(self.root), 6)
        (self.root / 'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            validate_completion(self.root)

    def test_stale_accuracy_and_missing_predictions_rejected(self):
        self.make_complete()
        write_json(self.root / 'accuracy.json', {'results': {'different': {'score': 1}}})
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_completion(self.root)
        self.make_complete()
        self.save_rows(self.rows[:3])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_completion(self.root)

    def test_gpu_guard_filters_only_authorized_physical_cards(self):
        gpu_csv = '0, GPU-zero\n2, GPU-two\n3, GPU-three\n'
        with patch('qdlm_accuracy_resume.subprocess.check_output', side_effect=[gpu_csv, 'GPU-zero, 123\n']):
            with contextlib.redirect_stdout(io.StringIO()):
                require_idle_gpus()
        with patch('qdlm_accuracy_resume.subprocess.check_output', side_effect=[gpu_csv, 'GPU-three, 456\n']):
            with self.assertRaisesRegex(RuntimeError, 'Launch cancelled'):
                require_idle_gpus()


if __name__ == '__main__':
    unittest.main()
