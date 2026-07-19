import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ddi.core import (
    QueryState,
    compute_retrieval_metrics,
    decode_selected_token_cues,
    mean_topk_overlap,
    rank_change_summary,
    select_disagreement_candidates,
    select_top_candidates,
)
from ddi.experiment import build_rstp_manifest, interact_once
from ddi.qwen_client import QwenVLClient, parse_json_object
from utils.simple_tokenizer import SimpleTokenizer


class FakeClient:
    def __init__(self):
        self.question_calls = 0
        self.answer_calls = 0

    def generate_question(self, query, candidates, gallery_paths, token_cues, method="ddi"):
        self.question_calls += 1
        return {
            "attribute": "bag",
            "question": "What kind of bag is the person carrying?",
            "cached": False,
            "request_model": "fake",
            "response_model": "fake",
            "usage": None,
        }

    def answer_question(self, question, source_image_path):
        self.answer_calls += 1
        return {
            "status": "confirmed",
            "fact": "The person carries a black backpack.",
            "cached": False,
            "request_model": "fake",
            "response_model": "fake",
            "usage": None,
        }


class FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "model": "qwen3.6-flash-2026-04-16",
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"total_tokens": 3},
        }


class DDISelectionTests(unittest.TestCase):
    def test_balanced_disagreement_selection(self):
        bge = [10, 9, 8, 7, 6, 5]
        tse = [5, 6, 7, 8, 9, 10]
        candidates = select_disagreement_candidates(bge, tse, k=5, m=4)
        self.assertEqual([item.gallery_index for item in candidates], [0, 1, 5, 4])
        self.assertEqual(
            [item.direction for item in candidates],
            ["bge_preferred", "bge_preferred", "tse_preferred", "tse_preferred"],
        )
        self.assertEqual(candidates[0].tse_rank, 6)
        self.assertEqual(candidates[2].bge_rank, 6)

    def test_direction_shortage_is_filled_deterministically(self):
        bge = [5, 4, 3, 2, 1]
        tse = [5, 4, 2, 3, 1]
        candidates = select_disagreement_candidates(bge, tse, k=5, m=4)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len({item.gallery_index for item in candidates}), 4)

    def test_joint_topk_tie_uses_gallery_index(self):
        candidates = select_top_candidates([1.0, 2.0, 2.0, 0.0], k=3)
        self.assertEqual([item.gallery_index for item in candidates], [1, 2, 0])


class QueryAndCueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = SimpleTokenizer()

    def test_selected_bpe_positions_become_words(self):
        text = "A person in a black jacket carries a red backpack."
        encoded = [self.tokenizer.encoder["<|startoftext|>"]]
        encoded += self.tokenizer.encode(text)
        encoded += [self.tokenizer.encoder["<|endoftext|>"]]
        encoded += [0] * (77 - len(encoded))
        selected = list(range(1, min(15, encoded.index(max(encoded)))))
        cues = decode_selected_token_cues(encoded, selected, self.tokenizer, max_cues=20)
        joined = " ".join(cues).lower()
        self.assertIn("black", joined)
        self.assertTrue("jacket" in joined or "backpack" in joined)

    def test_query_update_preserves_newest_fact_and_limit(self):
        original = " ".join(["person"] * 100)
        state = QueryState(original)
        state.add_fact("The person carries a black backpack", self.tokenizer, text_length=77)
        tokens = self.tokenizer.encode(state.current_query)
        self.assertLessEqual(len(tokens), 75)
        self.assertIn("black backpack", state.current_query.lower())
        self.assertFalse(state.add_fact("The person carries a black backpack.", self.tokenizer))


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics_and_rank_changes(self):
        similarity = np.array([[4, 3, 2, 1], [1, 2, 4, 3]], dtype=np.float32)
        metrics = compute_retrieval_metrics(similarity, [1, 2], [1, 1, 2, 2])
        self.assertAlmostEqual(metrics["rank1"], 100.0)
        self.assertAlmostEqual(metrics["mAP"], 100.0)
        self.assertEqual(metrics["target_ranks"], [1, 1])
        changes = rank_change_summary([3, 2, 1], [2, 2, 3])
        self.assertAlmostEqual(changes["improved"], 100.0 / 3.0)
        self.assertAlmostEqual(changes["unchanged"], 100.0 / 3.0)
        self.assertAlmostEqual(changes["worsened"], 100.0 / 3.0)

    def test_overlap(self):
        first = np.array([[4, 3, 2, 1]], dtype=np.float32)
        second = np.array([[4, 1, 3, 2]], dtype=np.float32)
        mean, per_query = mean_topk_overlap(first, second, k=2)
        self.assertEqual(per_query, [0.5])
        self.assertAlmostEqual(mean, 50.0)


class ClientTests(unittest.TestCase):
    def test_json_extraction(self):
        parsed = parse_json_object("```json\n{\"status\": \"uncertain\"}\n```")
        self.assertEqual(parsed["status"], "uncertain")

    def test_http_response_is_cached_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            client = QwenVLClient(
                base_url="https://example.invalid/v1",
                api_key="secret",
                cache_path=str(Path(directory) / "cache.json"),
                max_retries=0,
            )
            content = [{"type": "text", "text": "return json"}]
            with mock.patch("ddi.qwen_client.requests.post", return_value=FakeResponse()) as post:
                first = client._chat(content, namespace="test", max_tokens=8)
                second = client._chat(content, namespace="test", max_tokens=8)
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(post.call_count, 1)
            saved = json.loads((Path(directory) / "cache.json").read_text())
            self.assertNotIn("secret", json.dumps(saved))


class ManifestAndStateMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = SimpleTokenizer()

    def test_rstp_manifest_is_paired_and_reproducible(self):
        first = build_rstp_manifest("/root/datasets", seed=42)
        second = build_rstp_manifest("/root/datasets", seed=42)
        self.assertEqual(len(first["queries"]), 200)
        self.assertEqual(len(first["gallery"]), 1000)
        self.assertEqual(first["queries"], second["queries"])
        for query in first["queries"]:
            gallery = first["gallery"][query["source_gallery_index"]]
            self.assertEqual(query["pid"], gallery["pid"])
            self.assertEqual(query["source_image_path"], gallery["image_path"])

    def test_three_round_fake_client_flow(self):
        states = [QueryState("A person in dark clothing."), QueryState("A person in red.")]
        bge = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], dtype=np.float32)
        tse = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.float32)
        client = FakeClient()
        records = []
        for round_index in range(1, 4):
            records.extend(
                interact_once(
                    states,
                    bge,
                    tse,
                    [["clothing"], ["red"]],
                    client,
                    ["image"] * 5,
                    ["source1", "source2"],
                    self.tokenizer,
                    round_index=round_index,
                    method="ddi",
                    k=5,
                    m=4,
                )
            )
        self.assertEqual(len(records), 6)
        self.assertEqual(client.question_calls, 6)
        self.assertEqual(client.answer_calls, 6)
        self.assertIn("black backpack", states[0].current_query.lower())
        self.assertTrue(all(record["error"] is None for record in records))


if __name__ == "__main__":
    unittest.main()
