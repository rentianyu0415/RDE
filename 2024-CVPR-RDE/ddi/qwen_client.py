import base64
import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from PIL import Image

from .core import Candidate


PROMPT_VERSION = "ddi-qwen-v1"


class QwenAPIError(RuntimeError):
    def __init__(self, status_code: Optional[int], message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def parse_json_object(text: str) -> Dict[str, object]:
    if not isinstance(text, str):
        raise ValueError("model response must be text")
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model response")


def image_to_data_url(image_path: str, long_edge: int = 768, quality: int = 90) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = float(long_edge) / max(width, height)
        if scale != 1.0:
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


class ResponseCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                self.data = json.load(handle)
        else:
            self.data = {}

    def get(self, key: str) -> Optional[Dict[str, object]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, object]):
        self.data[key] = value
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
        os.replace(str(temporary), str(self.path))


class QwenVLClient:
    """Minimal OpenAI-compatible client for Qwen Flash vision models."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        model: str = "qwen3.6-flash-2026-04-16",
        fallback_model: str = "qwen3.6-flash",
        cache_path: str = "ddi_outputs/qwen_cache.json",
        timeout: float = 120.0,
        max_retries: int = 4,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.endpoint = self._normalize_endpoint(base_url)
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError("API key is missing; set {}".format(api_key_env))
        self.preferred_model = model
        self.fallback_model = fallback_model
        self.locked_model = None
        self.response_model = None
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = ResponseCache(cache_path)
        self.actual_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _normalize_endpoint(base_url: str) -> str:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return base_url + "/chat/completions"

    @property
    def model(self) -> str:
        return self.locked_model or self.preferred_model

    def generate_question(
        self,
        query: str,
        candidates: Sequence[Candidate],
        gallery_paths: Sequence[str],
        token_cues: Sequence[str],
        method: str = "ddi",
    ) -> Dict[str, object]:
        if method not in {"ddi", "joint_top5"}:
            raise ValueError("unknown interaction method: {}".format(method))

        if method == "ddi":
            metadata = [
                "Candidate {}: {}, BGE rank {}, TSE rank {}.".format(
                    offset + 1,
                    candidate.direction.replace("_", " "),
                    candidate.bge_rank,
                    candidate.tse_rank,
                )
                for offset, candidate in enumerate(candidates)
            ]
            cue_text = ", ".join(token_cues) if token_cues else "none"
            method_text = (
                "The candidates are selected because the global and fine-grained RDE "
                "branches disagree. Use the two preference groups and ranks to find one "
                "visible attribute that best separates them. TSE token cues: {}."
            ).format(cue_text)
        else:
            metadata = [
                "Candidate {} from the joint RDE Top-5.".format(offset + 1)
                for offset, _ in enumerate(candidates)
            ]
            method_text = (
                "These candidates come from the joint RDE Top-5. Compare them directly; "
                "no branch-specific evidence is available."
            )

        instruction = (
            "You generate exactly one short English question for interactive text-to-image "
            "person retrieval. Current query: {query}\n{method_text}\n"
            "Ask about exactly one visible person attribute not already confirmed in the query. "
            "Allowed attributes: clothing color or type, shoes, bag, hat, clothing pattern, "
            "handheld item, or a local accessory. Never ask about identity, background, camera, "
            "ranking, location, age, gender, ethnicity, or transient pose. Do not explain the "
            "RDE decision. Return only JSON: "
            '{{"attribute":"allowed attribute name","question":"short question"}}.'
        ).format(query=query, method_text=method_text)

        content = [{"type": "text", "text": instruction}]
        for label, candidate in zip(metadata, candidates):
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(gallery_paths[candidate.gallery_index])
                    },
                }
            )

        result = self._chat(content, namespace="question:{}".format(method), max_tokens=96)
        parsed = parse_json_object(result["content"])
        attribute = str(parsed.get("attribute", "")).strip()
        question = str(parsed.get("question", "")).strip()
        if not attribute or not question or len(question) > 240:
            raise ValueError("invalid question JSON returned by model")
        return {
            "attribute": attribute,
            "question": question,
            "cached": result["cached"],
            "request_model": result["request_model"],
            "response_model": result.get("response_model"),
            "usage": result.get("usage"),
        }

    def answer_question(self, question: str, source_image_path: str) -> Dict[str, object]:
        instruction = (
            "Act as a constrained user proxy for person retrieval. Inspect only the provided "
            "source image and answer only this English attribute question: {question}\n"
            "If the attribute is clearly visible, return one short English atomic fact, including "
            "a negative fact when absence is clearly visible. Do not add unasked details. If it is "
            "occluded, ambiguous, too small, or not visible, mark it uncertain. Return only JSON: "
            '{{"status":"confirmed|uncertain","fact":"one fact or empty string"}}.'
        ).format(question=question)
        content = [
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(source_image_path)},
            },
        ]
        result = self._chat(content, namespace="answer", max_tokens=64)
        parsed = parse_json_object(result["content"])
        status = str(parsed.get("status", "")).strip().lower()
        fact = str(parsed.get("fact", "")).strip()
        if status not in {"confirmed", "uncertain"}:
            raise ValueError("invalid answer status returned by model")
        if status == "confirmed" and not fact:
            raise ValueError("confirmed answer must contain a fact")
        if status == "uncertain":
            fact = ""
        return {
            "status": status,
            "fact": fact,
            "cached": result["cached"],
            "request_model": result["request_model"],
            "response_model": result.get("response_model"),
            "usage": result.get("usage"),
        }

    def _chat(
        self,
        content: List[Dict[str, object]],
        namespace: str,
        max_tokens: int,
    ) -> Dict[str, object]:
        key_payload = {
            "prompt_version": PROMPT_VERSION,
            "namespace": namespace,
            "model": self.model,
            "content": content,
            "max_tokens": max_tokens,
        }
        cache_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            cached = dict(cached)
            cached["cached"] = True
            cached_model = cached.get("request_model")
            if self.locked_model is None and cached_model:
                self.locked_model = cached_model
            if self.response_model is None:
                self.response_model = cached.get("response_model") or cached_model
            return cached

        response = self._request(content, max_tokens)
        response["cached"] = False
        self.cache.set(cache_key, response)
        if response.get("request_model") != key_payload["model"]:
            key_payload["model"] = response["request_model"]
            actual_key = hashlib.sha256(
                json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            self.cache.set(actual_key, response)
        return response

    def _request(self, content: List[Dict[str, object]], max_tokens: int) -> Dict[str, object]:
        models = [self.model]
        if self.locked_model is None and self.fallback_model not in models:
            models.append(self.fallback_model)

        last_error = None
        for model_index, model in enumerate(models):
            try:
                result = self._request_model(model, content, max_tokens)
            except QwenAPIError as error:
                last_error = error
                body = error.message.lower()
                model_missing = error.status_code in {400, 404} and any(
                    token in body
                    for token in (
                        "model_not_found",
                        "model not found",
                        "model does not exist",
                        "invalid model",
                        "unknown model",
                    )
                )
                if model_index == 0 and len(models) > 1 and model_missing:
                    continue
                raise
            self.locked_model = model
            self.response_model = result.get("response_model") or model
            return result
        raise last_error or QwenAPIError(None, "Qwen request failed")

    def _request_model(
        self, model: str, content: List[Dict[str, object]], max_tokens: int
    ) -> Dict[str, object]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "enable_thinking": False,
        }
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        unknown_parameter_retry = True

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                self.actual_calls += 1
            except requests.RequestException as error:
                if attempt >= self.max_retries:
                    raise QwenAPIError(None, str(error))
                self._backoff(attempt)
                continue

            if response.status_code >= 400:
                body = response.text[:4000]
                lower_body = body.lower()
                if (
                    response.status_code == 400
                    and unknown_parameter_retry
                    and "enable_thinking" in lower_body
                ):
                    payload.pop("enable_thinking", None)
                    unknown_parameter_retry = False
                    continue
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue
                raise QwenAPIError(response.status_code, body)

            try:
                data = response.json()
                message_content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise QwenAPIError(response.status_code, "invalid API response: {}".format(error))
            if isinstance(message_content, list):
                message_content = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in message_content
                )
            return {
                "content": str(message_content),
                "request_model": model,
                "response_model": data.get("model"),
                "usage": data.get("usage"),
                "created_at": int(time.time()),
            }

        raise QwenAPIError(None, "request retries exhausted")

    @staticmethod
    def _backoff(attempt: int):
        delay = min(30.0, (2 ** attempt) + random.random())
        time.sleep(delay)
