import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Candidate:
    gallery_index: int
    direction: str
    bge_rank: int
    tse_rank: int
    disagreement: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "gallery_index": self.gallery_index,
            "direction": self.direction,
            "bge_rank": self.bge_rank,
            "tse_rank": self.tse_rank,
            "disagreement": self.disagreement,
        }


def _topk_indices(scores: Sequence[float], k: int) -> List[int]:
    k = min(k, len(scores))
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:k]


def select_top_candidates(scores: Sequence[float], k: int = 5) -> List[Candidate]:
    indices = _topk_indices(scores, k)
    return [
        Candidate(index, "joint", rank, rank, 0.0)
        for rank, index in enumerate(indices, start=1)
    ]


def select_disagreement_candidates(
    bge_scores: Sequence[float],
    tse_scores: Sequence[float],
    k: int = 5,
    m: int = 4,
) -> List[Candidate]:
    """Select balanced BGE- and TSE-preferred candidates from two Top-K lists."""
    if k <= 0 or m <= 0:
        raise ValueError("k and m must be positive")
    if len(bge_scores) != len(tse_scores):
        raise ValueError("BGE and TSE scores must have the same length")

    bge_top = _topk_indices(bge_scores, k)
    tse_top = _topk_indices(tse_scores, k)
    bge_ranks = {index: rank for rank, index in enumerate(bge_top, start=1)}
    tse_ranks = {index: rank for rank, index in enumerate(tse_top, start=1)}
    missing_rank = k + 1
    union = sorted(set(bge_top).union(tse_top))

    candidates = []
    for index in union:
        bge_rank = bge_ranks.get(index, missing_rank)
        tse_rank = tse_ranks.get(index, missing_rank)
        signed = float(bge_rank - tse_rank) / float(k)
        if bge_rank < tse_rank:
            direction = "bge_preferred"
        elif tse_rank < bge_rank:
            direction = "tse_preferred"
        else:
            direction = "tie"
        candidates.append(
            Candidate(index, direction, bge_rank, tse_rank, signed)
        )

    def directional_key(candidate: Candidate) -> Tuple[float, int, int]:
        return (
            -abs(candidate.bge_rank - candidate.tse_rank),
            min(candidate.bge_rank, candidate.tse_rank),
            candidate.gallery_index,
        )

    bge_candidates = sorted(
        (item for item in candidates if item.direction == "bge_preferred"),
        key=directional_key,
    )
    tse_candidates = sorted(
        (item for item in candidates if item.direction == "tse_preferred"),
        key=directional_key,
    )

    bge_quota = m // 2
    tse_quota = m - bge_quota
    selected = bge_candidates[:bge_quota] + tse_candidates[:tse_quota]
    selected_indices = {item.gallery_index for item in selected}

    if len(selected) < min(m, len(candidates)):
        remaining = sorted(
            (item for item in candidates if item.gallery_index not in selected_indices),
            key=directional_key,
        )
        selected.extend(remaining[: min(m, len(candidates)) - len(selected)])

    return selected


def decode_selected_token_cues(
    token_ids: Sequence[int],
    selected_positions: Sequence[int],
    tokenizer,
    max_cues: int = 8,
) -> List[str]:
    """Map selected BPE positions back to readable words, ordered by attention rank."""
    tokens = [int(value) for value in token_ids]
    if not tokens:
        return []
    try:
        eot_position = tokens.index(max(tokens))
    except ValueError:
        eot_position = len(tokens)

    selected_rank = {}
    for rank, position in enumerate(selected_positions):
        position = int(position)
        if 0 < position < eot_position and position not in selected_rank:
            selected_rank[position] = rank

    words = []
    word_token_ids = []
    word_positions = []
    for position in range(1, eot_position):
        token_id = tokens[position]
        raw_piece = tokenizer.decoder.get(token_id, "")
        if raw_piece.startswith("<|"):
            continue
        word_token_ids.append(token_id)
        word_positions.append(position)
        if raw_piece.endswith("</w>"):
            if any(pos in selected_rank for pos in word_positions):
                cue = tokenizer.decode(word_token_ids).strip()
                cue = re.sub(r"\s+", " ", cue)
                if cue and re.search(r"[a-z0-9]", cue, flags=re.IGNORECASE):
                    rank = min(selected_rank[pos] for pos in word_positions if pos in selected_rank)
                    words.append((rank, cue))
            word_token_ids = []
            word_positions = []

    if word_token_ids and any(pos in selected_rank for pos in word_positions):
        cue = tokenizer.decode(word_token_ids).strip()
        if cue:
            rank = min(selected_rank[pos] for pos in word_positions if pos in selected_rank)
            words.append((rank, cue))

    result = []
    seen = set()
    for _, cue in sorted(words, key=lambda item: (item[0], item[1].lower())):
        normalized = cue.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(cue)
        if len(result) >= max_cues:
            break
    return result


def _normalize_fact(fact: str) -> str:
    fact = re.sub(r"\s+", " ", fact or "").strip()
    if fact and fact[-1] not in ".!?":
        fact += "."
    return fact


@dataclass
class QueryState:
    original_query: str
    facts: List[str] = field(default_factory=list)
    current_query: Optional[str] = None

    def __post_init__(self):
        self.original_query = re.sub(r"\s+", " ", self.original_query).strip()
        if self.current_query is None:
            self.current_query = self.original_query

    def add_fact(self, fact: str, tokenizer, text_length: int = 77) -> bool:
        fact = _normalize_fact(fact)
        if not fact:
            return False
        if fact.lower() in {item.lower() for item in self.facts}:
            return False
        self.facts.append(fact)
        self.current_query = self._fit_query(tokenizer, text_length)
        return True

    def _fit_query(self, tokenizer, text_length: int) -> str:
        max_content_tokens = max(1, text_length - 2)

        while len(self.facts) > 1:
            candidate = self._render(self.original_query, self.facts)
            if len(tokenizer.encode(candidate)) <= max_content_tokens:
                return candidate
            self.facts.pop(0)

        suffix = ""
        if self.facts:
            suffix = " Additional detail: " + " ".join(self.facts)
        suffix_tokens = tokenizer.encode(suffix)
        if len(suffix_tokens) >= max_content_tokens:
            return tokenizer.decode(suffix_tokens[:max_content_tokens]).strip()

        original_budget = max_content_tokens - len(suffix_tokens)
        original_tokens = tokenizer.encode(self.original_query)[:original_budget]
        original = tokenizer.decode(original_tokens).strip()
        return self._render(original, self.facts)

    @staticmethod
    def _render(original: str, facts: Sequence[str]) -> str:
        if not facts:
            return original.strip()
        return (original.strip() + " Additional detail: " + " ".join(facts)).strip()

    def as_dict(self) -> Dict[str, object]:
        return {
            "original_query": self.original_query,
            "facts": list(self.facts),
            "current_query": self.current_query,
        }


def compute_retrieval_metrics(
    similarity: np.ndarray,
    query_pids: Sequence[int],
    gallery_pids: Sequence[int],
) -> Dict[str, object]:
    similarity = np.asarray(similarity)
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    if similarity.shape != (len(query_pids), len(gallery_pids)):
        raise ValueError("similarity shape does not match query/gallery labels")

    indices = np.argsort(-similarity, axis=1, kind="stable")
    matches = gallery_pids[indices] == query_pids[:, None]
    if not np.all(matches.any(axis=1)):
        raise ValueError("each query must have at least one positive gallery image")

    cumulative = np.cumsum(matches, axis=1)
    cmc = np.minimum(cumulative, 1).mean(axis=0) * 100.0
    precisions = cumulative / (np.arange(matches.shape[1])[None, :] + 1.0)
    average_precision = (precisions * matches).sum(axis=1) / matches.sum(axis=1)
    target_ranks = np.argmax(matches, axis=1) + 1

    inverse_negative_penalties = []
    for row_index, row in enumerate(matches):
        last_positive = np.flatnonzero(row)[-1]
        inverse_negative_penalties.append(
            cumulative[row_index, last_positive] / float(last_positive + 1)
        )

    def cmc_at(rank: int) -> float:
        return float(cmc[min(rank, len(cmc)) - 1])

    return {
        "rank1": cmc_at(1),
        "rank5": cmc_at(5),
        "rank10": cmc_at(10),
        "mAP": float(average_precision.mean() * 100.0),
        "mINP": float(np.mean(inverse_negative_penalties) * 100.0),
        "target_ranks": target_ranks.tolist(),
        "indices": indices,
    }


def mean_topk_overlap(
    first_similarity: np.ndarray,
    second_similarity: np.ndarray,
    k: int = 5,
) -> Tuple[float, List[float]]:
    first = np.argsort(-np.asarray(first_similarity), axis=1, kind="stable")[:, :k]
    second = np.argsort(-np.asarray(second_similarity), axis=1, kind="stable")[:, :k]
    values = [len(set(a).intersection(b)) / float(k) for a, b in zip(first, second)]
    return float(np.mean(values) * 100.0), values


def rank_change_summary(
    before: Sequence[int], after: Sequence[int]
) -> Dict[str, float]:
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        raise ValueError("rank vectors must have the same shape")
    count = max(1, len(before))
    return {
        "improved": float(np.sum(after < before) * 100.0 / count),
        "unchanged": float(np.sum(after == before) * 100.0 / count),
        "worsened": float(np.sum(after > before) * 100.0 / count),
    }
