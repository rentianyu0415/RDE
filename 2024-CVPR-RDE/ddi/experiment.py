import csv
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.bases import ImageDataset, tokenize
from datasets.build import build_transforms
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer

from .core import (
    QueryState,
    compute_retrieval_metrics,
    decode_selected_token_cues,
    mean_topk_overlap,
    rank_change_summary,
    select_disagreement_candidates,
    select_top_candidates,
)
from .qwen_client import PROMPT_VERSION


LOGGER = logging.getLogger("RDE.DDI")


def sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_rstp_manifest(root_dir: str, seed: int = 42) -> Dict[str, object]:
    dataset_dir = Path(root_dir) / "RSTPReid"
    annotation_path = dataset_dir / "data_captions.json"
    image_dir = dataset_dir / "imgs"
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    test_annotations = [item for item in annotations if item["split"] == "test"]
    gallery = []
    pairs_by_pid = {}
    for gallery_index, annotation in enumerate(test_annotations):
        pid = int(annotation["id"])
        image_path = str(image_dir / annotation["img_path"])
        gallery.append(
            {
                "gallery_index": gallery_index,
                "pid": pid,
                "image_path": image_path,
                "relative_image_path": annotation["img_path"],
            }
        )
        for caption_index, caption in enumerate(annotation["captions"]):
            pairs_by_pid.setdefault(pid, []).append(
                {
                    "pid": pid,
                    "caption": caption,
                    "caption_index": caption_index,
                    "source_image_path": image_path,
                    "source_gallery_index": gallery_index,
                    "relative_image_path": annotation["img_path"],
                }
            )

    rng = random.Random(seed)
    queries = []
    for query_index, pid in enumerate(sorted(pairs_by_pid)):
        pairs = sorted(
            pairs_by_pid[pid],
            key=lambda item: (item["relative_image_path"], item["caption_index"]),
        )
        selected = dict(rng.choice(pairs))
        selected["query_index"] = query_index
        queries.append(selected)

    if len(queries) != 200 or len({item["pid"] for item in queries}) != 200:
        raise ValueError("RSTPReid test split must contain exactly 200 identities")
    if len(gallery) != 1000:
        raise ValueError("RSTPReid test gallery must contain exactly 1000 images")
    for query in queries:
        source = gallery[query["source_gallery_index"]]
        if source["pid"] != query["pid"] or source["image_path"] != query["source_image_path"]:
            raise ValueError("query caption/source image pairing is inconsistent")

    return {
        "dataset": "RSTPReid",
        "seed": seed,
        "gallery": gallery,
        "queries": queries,
    }


def _json_safe_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in metrics.items() if key != "indices"}


def evaluate_similarities(
    bge_similarity: np.ndarray,
    tse_similarity: np.ndarray,
    query_pids: Sequence[int],
    gallery_pids: Sequence[int],
    previous_target_ranks: Optional[Sequence[int]] = None,
    baseline_target_ranks: Optional[Sequence[int]] = None,
    overlap_k: int = 5,
) -> Dict[str, object]:
    joint_similarity = (bge_similarity + tse_similarity) / 2.0
    bge_metrics = compute_retrieval_metrics(bge_similarity, query_pids, gallery_pids)
    tse_metrics = compute_retrieval_metrics(tse_similarity, query_pids, gallery_pids)
    joint_metrics = compute_retrieval_metrics(joint_similarity, query_pids, gallery_pids)
    overlap, overlap_per_query = mean_topk_overlap(
        bge_similarity, tse_similarity, k=overlap_k
    )
    result = {
        "BGE": _json_safe_metrics(bge_metrics),
        "TSE": _json_safe_metrics(tse_metrics),
        "BGE+TSE": _json_safe_metrics(joint_metrics),
        "overlap_at_{}".format(overlap_k): overlap,
        "overlap_per_query": overlap_per_query,
    }
    target_ranks = joint_metrics["target_ranks"]
    if previous_target_ranks is not None:
        result["rank_change_vs_previous"] = rank_change_summary(
            previous_target_ranks, target_ranks
        )
    if baseline_target_ranks is not None:
        result["rank_change_vs_baseline"] = rank_change_summary(
            baseline_target_ranks, target_ranks
        )
    return result


def interact_once(
    states: Sequence[QueryState],
    bge_similarity: np.ndarray,
    tse_similarity: np.ndarray,
    token_cues: Sequence[Sequence[str]],
    client,
    gallery_paths: Sequence[str],
    source_paths: Sequence[str],
    tokenizer,
    round_index: int,
    method: str = "ddi",
    k: int = 5,
    m: int = 4,
    text_length: int = 77,
) -> List[Dict[str, object]]:
    joint_similarity = (bge_similarity + tse_similarity) / 2.0
    records = []
    for query_index, state in enumerate(states):
        if method == "ddi":
            candidates = select_disagreement_candidates(
                bge_similarity[query_index], tse_similarity[query_index], k=k, m=m
            )
        elif method == "joint_top5":
            candidates = select_top_candidates(joint_similarity[query_index], k=k)
        else:
            raise ValueError("unknown interaction method: {}".format(method))

        before_query = state.current_query
        record = {
            "query_index": query_index,
            "round": round_index,
            "method": method,
            "query_before": before_query,
            "token_cues": list(token_cues[query_index]) if method == "ddi" else [],
            "candidates": [candidate.as_dict() for candidate in candidates],
            "question": None,
            "answer": None,
            "query_after": before_query,
            "error": None,
        }
        try:
            question = client.generate_question(
                before_query,
                candidates,
                gallery_paths,
                token_cues[query_index] if method == "ddi" else [],
                method=method,
            )
            record["question"] = question
            answer = client.answer_question(question["question"], source_paths[query_index])
            record["answer"] = answer
            if answer["status"] == "confirmed":
                state.add_fact(answer["fact"], tokenizer, text_length=text_length)
            record["query_after"] = state.current_query
        except Exception as error:  # keep one failed query from aborting an expensive run
            record["error"] = "{}: {}".format(type(error).__name__, error)
            LOGGER.exception("interaction failed for query %s", query_index)
        records.append(record)
        if (query_index + 1) % 10 == 0 or query_index + 1 == len(states):
            LOGGER.info(
                "%s round %s: completed %s/%s queries",
                method,
                round_index,
                query_index + 1,
                len(states),
            )
    return records


def summarize_interactions(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    count = len(records)
    errors = sum(record.get("error") is not None for record in records)
    confirmed = sum(
        (record.get("answer") or {}).get("status") == "confirmed" for record in records
    )
    uncertain = sum(
        (record.get("answer") or {}).get("status") == "uncertain" for record in records
    )
    denominator = max(1, count)
    return {
        "queries": count,
        "nominal_calls": count * 2,
        "nominal_calls_per_query": 2 if count else 0,
        "errors": errors,
        "error_rate": errors * 100.0 / denominator,
        "confirmed": confirmed,
        "uncertain": uncertain,
        "uncertain_rate": uncertain * 100.0 / denominator,
    }


class DDIExperiment:
    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        root_dir: str,
        output_dir: str,
        client,
        device: str = "cuda",
        seed: int = 42,
        batch_size: int = 256,
        num_workers: int = 8,
        k: int = 5,
        m: int = 4,
        rounds: int = 3,
    ):
        self.config_file = str(Path(config_file).resolve())
        self.checkpoint_file = str(Path(checkpoint_file).resolve())
        self.root_dir = str(Path(root_dir).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this process")
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k
        self.m = m
        self.rounds = rounds
        self.tokenizer = SimpleTokenizer()
        self.checkpoint_sha256 = sha256_file(self.checkpoint_file)
        self.manifest = build_rstp_manifest(self.root_dir, seed=self.seed)
        self.query_pids = [item["pid"] for item in self.manifest["queries"]]
        self.gallery_pids = [item["pid"] for item in self.manifest["gallery"]]
        self.gallery_paths = [item["image_path"] for item in self.manifest["gallery"]]
        self.source_paths = [item["source_image_path"] for item in self.manifest["queries"]]
        self.original_queries = [item["caption"] for item in self.manifest["queries"]]
        self.model, self.args = self._load_model()
        self.gallery_bge, self.gallery_tse = self._load_or_compute_gallery_features()
        self.main_result = None
        self.ablation_result = None

    def _load_model(self):
        args = load_train_configs(self.config_file)
        args.training = False
        args.root_dir = self.root_dir
        args.test_batch_size = self.batch_size
        args.num_workers = self.num_workers
        self.dataset = RSTPReid(root=self.root_dir, verbose=False)
        model = build_model(args, len(self.dataset.train_id_container))
        Checkpointer(model).load(f=self.checkpoint_file)
        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, args

    def _load_or_compute_gallery_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self.output_dir / "gallery_features.pt"
        expected = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "gallery_paths": self.gallery_paths,
            "img_size": list(self.args.img_size),
            "select_ratio": float(self.args.select_ratio),
        }
        if cache_path.exists():
            cached = torch.load(str(cache_path), map_location="cpu")
            if cached.get("metadata") == expected:
                LOGGER.info("loaded cached gallery features from %s", cache_path)
                return cached["bge"], cached["tse"]

        transform = build_transforms(img_size=self.args.img_size, is_train=False)
        dataset = ImageDataset(self.gallery_pids, self.gallery_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        bge_features = []
        tse_features = []
        with torch.no_grad():
            for _, images in loader:
                images = images.to(self.device)
                bge, tse = self.model.encode_image_dual(images)
                bge_features.append(F.normalize(bge.float(), p=2, dim=1).cpu())
                tse_features.append(F.normalize(tse.float(), p=2, dim=1).cpu())
        bge_features = torch.cat(bge_features, dim=0)
        tse_features = torch.cat(tse_features, dim=0)
        torch.save(
            {"metadata": expected, "bge": bge_features, "tse": tse_features},
            str(cache_path),
        )
        return bge_features, tse_features

    def encode_queries(
        self, queries: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        all_tokens = torch.stack(
            [
                tokenize(
                    query,
                    tokenizer=self.tokenizer,
                    text_length=self.args.text_length,
                    truncate=True,
                )
                for query in queries
            ]
        )
        bge_features = []
        tse_features = []
        selections = []
        with torch.no_grad():
            for start in range(0, len(all_tokens), self.batch_size):
                token_batch = all_tokens[start : start + self.batch_size].to(self.device)
                bge, tse, selected = self.model.encode_text_dual(
                    token_batch, return_selection=True
                )
                bge_features.append(F.normalize(bge.float(), p=2, dim=1).cpu())
                tse_features.append(F.normalize(tse.float(), p=2, dim=1).cpu())
                selections.append(selected.cpu())
        bge_features = torch.cat(bge_features, dim=0)
        tse_features = torch.cat(tse_features, dim=0)
        selections = torch.cat(selections, dim=0)
        cues = [
            decode_selected_token_cues(
                all_tokens[index].tolist(),
                selections[index].tolist(),
                self.tokenizer,
            )
            for index in range(len(all_tokens))
        ]
        bge_similarity = (bge_features @ self.gallery_bge.t()).numpy()
        tse_similarity = (tse_features @ self.gallery_tse.t()).numpy()
        return bge_similarity, tse_similarity, cues

    def verify_full_baseline(self, tolerance: float = 0.02) -> Dict[str, object]:
        captions = self.dataset.test["captions"]
        caption_pids = self.dataset.test["caption_pids"]
        bge_similarity, tse_similarity, _ = self.encode_queries(captions)
        result = evaluate_similarities(
            bge_similarity,
            tse_similarity,
            caption_pids,
            self.gallery_pids,
            overlap_k=self.k,
        )
        expected = {"rank1": 65.55, "rank5": 84.00, "rank10": 89.35, "mAP": 51.09}
        observed = result["BGE+TSE"]
        differences = {key: abs(observed[key] - value) for key, value in expected.items()}
        report = {
            "expected": expected,
            "observed": {key: observed[key] for key in expected},
            "absolute_differences": differences,
            "tolerance": tolerance,
            "passed": all(value <= tolerance for value in differences.values()),
        }
        self._write_json(self.output_dir / "baseline_regression.json", report)
        if not report["passed"]:
            raise RuntimeError("full-test RDE baseline regression exceeded tolerance: {}".format(report))
        LOGGER.info("full-test RDE baseline regression passed: %s", report["observed"])
        return report

    def preflight(
        self,
        bge_similarity: np.ndarray,
        tse_similarity: np.ndarray,
        cues: Sequence[Sequence[str]],
    ):
        candidates = select_disagreement_candidates(
            bge_similarity[0], tse_similarity[0], k=self.k, m=self.m
        )
        question = self.client.generate_question(
            self.original_queries[0],
            candidates,
            self.gallery_paths,
            cues[0],
            method="ddi",
        )
        answer = self.client.answer_question(question["question"], self.source_paths[0])
        if answer["status"] not in {"confirmed", "uncertain"}:
            raise RuntimeError("Qwen preflight returned an invalid answer")
        self._lock_experiment()
        return {"question": question, "answer": answer}

    def _lock_experiment(self, require_existing: bool = False):
        path = self.output_dir / "experiment_lock.json"
        previous = None
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            if self.client.locked_model is None:
                self.client.locked_model = previous["model"]
        elif require_existing:
            raise RuntimeError("--skip-preflight is only valid when an experiment lock already exists")

        lock = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "endpoint": self.client.endpoint,
            "model": self.client.model,
            "prompt_version": PROMPT_VERSION,
            "seed": self.seed,
            "k": self.k,
            "m": self.m,
        }
        if previous is not None:
            if previous != lock:
                raise RuntimeError(
                    "experiment lock mismatch; use a new output directory to avoid mixed results"
                )
        else:
            self._write_json(path, lock)
        manifest_experiment = dict(lock)
        manifest_experiment["response_model"] = self.client.response_model
        self.manifest["experiment"] = manifest_experiment
        self._write_json(self.output_dir / "manifest.json", self.manifest)

    def run_main(self, rounds: Optional[int] = None, do_preflight: bool = True):
        rounds = self.rounds if rounds is None else rounds
        states = [QueryState(query) for query in self.original_queries]
        bge_similarity, tse_similarity, cues = self.encode_queries(
            [state.current_query for state in states]
        )
        if do_preflight:
            preflight = self.preflight(bge_similarity, tse_similarity, cues)
        else:
            self._lock_experiment(require_existing=True)
            preflight = None

        summaries = []
        trajectories = []
        baseline = evaluate_similarities(
            bge_similarity,
            tse_similarity,
            self.query_pids,
            self.gallery_pids,
            overlap_k=self.k,
        )
        baseline["round"] = 0
        baseline["interaction"] = summarize_interactions([])
        summaries.append(baseline)
        LOGGER.info(
            "subset baseline: R1 %.2f, mAP %.2f, Overlap@%s %.2f",
            baseline["BGE+TSE"]["rank1"],
            baseline["BGE+TSE"]["mAP"],
            self.k,
            baseline["overlap_at_{}".format(self.k)],
        )
        baseline_ranks = baseline["BGE+TSE"]["target_ranks"]
        previous_ranks = baseline_ranks

        for round_index in range(1, rounds + 1):
            LOGGER.info("starting DDI round %s/%s", round_index, rounds)
            records = interact_once(
                states,
                bge_similarity,
                tse_similarity,
                cues,
                self.client,
                self.gallery_paths,
                self.source_paths,
                self.tokenizer,
                round_index=round_index,
                method="ddi",
                k=self.k,
                m=self.m,
                text_length=self.args.text_length,
            )
            bge_similarity, tse_similarity, cues = self.encode_queries(
                [state.current_query for state in states]
            )
            summary = evaluate_similarities(
                bge_similarity,
                tse_similarity,
                self.query_pids,
                self.gallery_pids,
                previous_target_ranks=previous_ranks,
                baseline_target_ranks=baseline_ranks,
                overlap_k=self.k,
            )
            summary["round"] = round_index
            summary["interaction"] = summarize_interactions(records)
            current_ranks = summary["BGE+TSE"]["target_ranks"]
            for record in records:
                index = record["query_index"]
                record["target_rank_before"] = previous_ranks[index]
                record["target_rank_after"] = current_ranks[index]
            trajectories.extend(records)
            summaries.append(summary)
            LOGGER.info(
                "DDI round %s: R1 %.2f, mAP %.2f, Overlap@%s %.2f",
                round_index,
                summary["BGE+TSE"]["rank1"],
                summary["BGE+TSE"]["mAP"],
                self.k,
                summary["overlap_at_{}".format(self.k)],
            )
            previous_ranks = current_ranks
            self._write_jsonl(self.output_dir / "main_trajectories.jsonl", trajectories)
            self._write_json(
                self.output_dir / "main_summary.json",
                {
                    "preflight": preflight,
                    "rounds": summaries,
                    "client": self._client_stats(),
                },
            )

        self.main_result = {
            "preflight": preflight,
            "rounds": summaries,
            "trajectories": trajectories,
            "client": self._client_stats(),
        }
        self.write_reports()
        return self.main_result

    def run_ablation(self, do_preflight: bool = True):
        states = [QueryState(query) for query in self.original_queries]
        bge_similarity, tse_similarity, cues = self.encode_queries(self.original_queries)
        if do_preflight:
            self.preflight(bge_similarity, tse_similarity, cues)
        else:
            self._lock_experiment(require_existing=True)
        baseline = evaluate_similarities(
            bge_similarity,
            tse_similarity,
            self.query_pids,
            self.gallery_pids,
            overlap_k=self.k,
        )
        records = interact_once(
            states,
            bge_similarity,
            tse_similarity,
            cues,
            self.client,
            self.gallery_paths,
            self.source_paths,
            self.tokenizer,
            round_index=1,
            method="joint_top5",
            k=self.k,
            m=self.m,
            text_length=self.args.text_length,
        )
        updated_bge, updated_tse, _ = self.encode_queries(
            [state.current_query for state in states]
        )
        summary = evaluate_similarities(
            updated_bge,
            updated_tse,
            self.query_pids,
            self.gallery_pids,
            previous_target_ranks=baseline["BGE+TSE"]["target_ranks"],
            baseline_target_ranks=baseline["BGE+TSE"]["target_ranks"],
            overlap_k=self.k,
        )
        summary["interaction"] = summarize_interactions(records)
        current_ranks = summary["BGE+TSE"]["target_ranks"]
        before_ranks = baseline["BGE+TSE"]["target_ranks"]
        for record in records:
            index = record["query_index"]
            record["target_rank_before"] = before_ranks[index]
            record["target_rank_after"] = current_ranks[index]
        self.ablation_result = {
            "baseline": baseline,
            "joint_top5": summary,
            "trajectories": records,
            "client": self._client_stats(),
        }
        self._write_jsonl(self.output_dir / "ablation_trajectories.jsonl", records)
        self._write_json(self.output_dir / "ablation_summary.json", self.ablation_result)
        self.write_reports()
        return self.ablation_result

    def _client_stats(self):
        return {
            "model": self.client.model,
            "response_model": self.client.response_model,
            "actual_http_calls": self.client.actual_calls,
            "cache_hits": self.client.cache_hits,
        }

    def write_reports(self):
        if self.main_result:
            table4 = [
                "| Method | Round | Rank-1 | Rank-5 | Rank-10 | mAP | Overlap@{} | MLLM calls/query |".format(self.k),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for item in self.main_result["rounds"]:
                metrics = item["BGE+TSE"]
                round_index = item["round"]
                method = "RDE" if round_index == 0 else "RDE + DDI"
                table4.append(
                    "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
                        method,
                        round_index,
                        metrics["rank1"],
                        metrics["rank5"],
                        metrics["rank10"],
                        metrics["mAP"],
                        item["overlap_at_{}".format(self.k)],
                        round_index * 2,
                    )
                )
            (self.output_dir / "table4.md").write_text("\n".join(table4) + "\n", encoding="utf-8")
            self._write_metrics_csv(self.main_result["rounds"])
            self._write_qualitative_report(self.main_result["trajectories"])

        if self.ablation_result:
            ddi_round = None
            if self.main_result and len(self.main_result["rounds"]) > 1:
                ddi_round = self.main_result["rounds"][1]
            lines = [
                "| Interaction input | Dual rankings | TSE cues | Rank-1 | mAP | Overlap@{} |".format(self.k),
                "|---|:---:|:---:|---:|---:|---:|",
            ]
            joint = self.ablation_result["joint_top5"]
            lines.append(
                "| Joint ranking Top-5 |  |  | {:.2f} | {:.2f} | {:.2f} |".format(
                    joint["BGE+TSE"]["rank1"],
                    joint["BGE+TSE"]["mAP"],
                    joint["overlap_at_{}".format(self.k)],
                )
            )
            if ddi_round is not None:
                lines.append(
                    "| BGE-TSE disagreement candidates | ✓ | ✓ | {:.2f} | {:.2f} | {:.2f} |".format(
                        ddi_round["BGE+TSE"]["rank1"],
                        ddi_round["BGE+TSE"]["mAP"],
                        ddi_round["overlap_at_{}".format(self.k)],
                    )
                )
            (self.output_dir / "table5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_metrics_csv(self, rounds: Sequence[Dict[str, object]]):
        path = self.output_dir / "round_metrics.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["round", "branch", "rank1", "rank5", "rank10", "mAP", "mINP", "overlap_at_{}".format(self.k)]
            )
            for item in rounds:
                for branch in ("BGE", "TSE", "BGE+TSE"):
                    metrics = item[branch]
                    writer.writerow(
                        [
                            item["round"],
                            branch,
                            metrics["rank1"],
                            metrics["rank5"],
                            metrics["rank10"],
                            metrics["mAP"],
                            metrics["mINP"],
                            item["overlap_at_{}".format(self.k)],
                        ]
                    )

    def _write_qualitative_report(self, trajectories: Sequence[Dict[str, object]]):
        if not trajectories:
            return
        scored = sorted(
            trajectories,
            key=lambda item: item.get("target_rank_before", 0) - item.get("target_rank_after", 0),
            reverse=True,
        )
        improved = [
            item for item in scored
            if item.get("target_rank_after", 0) < item.get("target_rank_before", 0)
        ][:2]
        unchanged = [
            item for item in scored
            if item.get("target_rank_after", 0) == item.get("target_rank_before", 0)
        ][:1]
        worsened = [
            item for item in reversed(scored)
            if item.get("target_rank_after", 0) > item.get("target_rank_before", 0)
        ][:2]
        examples = improved + unchanged + worsened
        if not examples:
            examples = scored[:5]
        lines = ["# DDI qualitative examples", ""]
        for item in examples:
            query = self.manifest["queries"][item["query_index"]]
            lines.extend(
                [
                    "## Query {} / round {}".format(item["query_index"], item["round"]),
                    "",
                    "- PID: `{}`".format(query["pid"]),
                    "- Source image: `{}`".format(query["relative_image_path"]),
                    "- Rank: {} → {}".format(item.get("target_rank_before"), item.get("target_rank_after")),
                    "- Before: {}".format(item["query_before"]),
                    "- TSE cues: {}".format(", ".join(item["token_cues"])),
                    "- Question: {}".format((item.get("question") or {}).get("question", "failed")),
                    "- Answer: {}".format((item.get("answer") or {}).get("fact", item.get("error", ""))),
                    "- After: {}".format(item["query_after"]),
                    "",
                ]
            )
        (self.output_dir / "qualitative.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, value: object):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(str(temporary), str(path))

    @staticmethod
    def _write_jsonl(path: Path, values: Sequence[Dict[str, object]]):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        os.replace(str(temporary), str(path))
