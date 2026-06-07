from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.private.json"


def mask_key(key: str) -> str:
    if not key:
        return "(empty)"
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:5] + "***" + key[-4:]


def endpoint(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def check_target(target: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": target["model"],
        "messages": [
            {"role": "system", "content": "请用最短答案完成联通性测试。"},
            {"role": "user", "content": "请只回复：OK"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    if target.get("reasoning_effort"):
        payload["reasoning_effort"] = target["reasoning_effort"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {target.get('api_key', '')}",
    }
    request = urllib.request.Request(
        endpoint(target["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=int(target.get("timeout", 180) or 180)) as response:
            raw = response.read().decode("utf-8", errors="replace")
            latency_ms = round((time.perf_counter() - started) * 1000)
            data = json.loads(raw)
            text = extract_text(data)
            return {
                "name": target["name"],
                "model": target["model"],
                "base_url": target["base_url"],
                "api_key": mask_key(target.get("api_key", "")),
                "reasoning_effort": target.get("reasoning_effort", ""),
                "ok": True,
                "status": response.status,
                "latency_ms": latency_ms,
                "reply": text[:120],
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "name": target["name"],
            "model": target["model"],
            "base_url": target["base_url"],
            "api_key": mask_key(target.get("api_key", "")),
            "reasoning_effort": target.get("reasoning_effort", ""),
            "ok": False,
            "status": exc.code,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "reply": "",
            "error": body[:500],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": target["name"],
            "model": target["model"],
            "base_url": target["base_url"],
            "api_key": mask_key(target.get("api_key", "")),
            "reasoning_effort": target.get("reasoning_effort", ""),
            "ok": False,
            "status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "reply": "",
            "error": str(exc),
        }


def extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        if isinstance(choices[0].get("text"), str):
            return choices[0]["text"].strip()
    return json.dumps(data, ensure_ascii=False)[:120]


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    results = [check_target(target) for target in config["targets"]]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
