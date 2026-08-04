import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_ENDPOINT = "https://api.siliconflow.cn/v1/rerank"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def default_instruction(model_name: str):
    if model_name.startswith("Qwen/"):
        return (
            "Rank the candidate event types by how likely the input text expresses the "
            "event type described by each schema. Focus on semantic match, not only "
            "surface word overlap."
        )
    return None


class SiliconFlowRerankerClient:
    def __init__(self, api_key: str, endpoint: str = DEFAULT_ENDPOINT, cache_dir: str | None = None):
        self.api_key = api_key
        self.endpoint = endpoint
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, model: str, payload: dict):
        if not self.cache_dir:
            return None
        payload_text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        model_dir = self.cache_dir / sanitize_model_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{digest}.json"

    def rerank(self, model: str, query: str, documents: list[str], top_n: int, instruction: str | None = None, retries: int = 3):
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        if instruction:
            payload["instruction"] = instruction

        cache_path = self._cache_path(model, payload)
        if cache_path and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                return data
            except urllib.error.HTTPError as exc:
                last_error = exc.read().decode("utf-8", errors="ignore")
                if exc.code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"SiliconFlow rerank failed with HTTP {exc.code}: {last_error}") from exc
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"SiliconFlow rerank failed: {last_error}") from exc

        raise RuntimeError(f"SiliconFlow rerank failed after retries: {last_error}")


def parse_rerank_results(response_json: dict):
    raw_results = response_json.get("results")
    if raw_results is None:
        raw_results = response_json.get("data")
    if raw_results is None:
        raise ValueError(f"Unexpected rerank response keys: {sorted(response_json.keys())}")

    parsed = []
    for item in raw_results:
        index = item.get("index", item.get("document_index"))
        score = item.get("relevance_score", item.get("score"))
        parsed.append({"index": index, "score": score})
    parsed.sort(key=lambda x: x["score"], reverse=True)
    return parsed


def load_api_key(env_name: str = "SILICONFLOW_API_KEY"):
    api_key = os.getenv(env_name)
    if api_key:
        return api_key

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_name:
                value = value.strip().strip("'").strip('"')
                if value:
                    return value

    raise ValueError(f"Missing API key in environment variable {env_name} and .env")
