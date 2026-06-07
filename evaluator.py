from __future__ import annotations

import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from html import escape
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "datasets" / "default_zh.json"
DATASET_DIR = ROOT / "datasets"
REPORT_DIR = ROOT / "reports"
PRIVATE_CONFIG_PATH = ROOT / "config.private.json"


@dataclass(frozen=True)
class ModelTarget:
    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: int = 360
    concurrency: int = 3
    reasoning_effort: str = ""

    @property
    def endpoint(self) -> str:
        cleaned = self.base_url.rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        if cleaned.endswith("/v1"):
            return f"{cleaned}/chat/completions"
        return f"{cleaned}/v1/chat/completions"


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_datasets() -> list[dict[str, Any]]:
    datasets = []
    for path in sorted(DATASET_DIR.glob("*.json")):
        dataset = load_dataset(path)
        datasets.append(
            {
                "id": path.stem,
                "name": dataset.get("name", path.stem),
                "version": dataset.get("version", ""),
                "description": dataset.get("description", ""),
                "case_count": len(dataset.get("cases", [])),
                "categories": sorted({case.get("category", "") for case in dataset.get("cases", [])}),
                "cases": [
                    {
                        "id": case.get("id", ""),
                        "category": case.get("category", ""),
                        "difficulty": case.get("difficulty", ""),
                        "prompt": case.get("prompt", ""),
                        "weight": case.get("weight", 1),
                    }
                    for case in dataset.get("cases", [])
                ],
            }
        )
    return datasets


def dataset_path_from_id(dataset_id: str | None) -> Path:
    if not dataset_id:
        return DATASET_PATH
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", dataset_id)
    path = DATASET_DIR / f"{safe_id}.json"
    if not path.exists():
        raise ValueError(f"测试集不存在：{dataset_id}")
    return path


def load_private_targets(path: Path = PRIVATE_CONFIG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    config = json.loads(path.read_text(encoding="utf-8"))
    targets = config.get("targets", [])
    if not isinstance(targets, list):
        raise ValueError("config.private.json 中的 targets 必须是数组。")
    return targets


def load_private_judge(path: Path = PRIVATE_CONFIG_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    judge = config.get("judge")
    return judge if isinstance(judge, dict) else None


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:5] + "***" + key[-4:]


def public_private_judge() -> dict[str, Any] | None:
    judge = load_private_judge()
    if not judge:
        return None
    return {
        "name": judge.get("name", "裁判模型"),
        "model": judge.get("model", ""),
        "base_url": judge.get("base_url", ""),
        "api_key": mask_key(str(judge.get("api_key", ""))),
        "has_server_key": bool(judge.get("api_key")),
        "concurrency": judge.get("concurrency", 1),
        "timeout": judge.get("timeout", 360),
        "reasoning_effort": judge.get("reasoning_effort", ""),
    }


def public_private_targets() -> list[dict[str, Any]]:
    public_targets = []
    for index, target in enumerate(load_private_targets(), start=1):
        public_targets.append(
            {
                "config_id": str(index),
                "name": target.get("name", f"模型 {index}"),
                "model": target.get("model", ""),
                "base_url": target.get("base_url", ""),
                "api_key": mask_key(str(target.get("api_key", ""))),
                "has_server_key": bool(target.get("api_key")),
                "concurrency": target.get("concurrency", 3),
                "timeout": target.get("timeout", 360),
                "reasoning_effort": target.get("reasoning_effort", ""),
            }
        )
    return public_targets


def resolve_config_targets(raw_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    private_targets = load_private_targets()
    resolved = []
    for raw in raw_targets:
        config_id = str(raw.get("config_id", "")).strip()
        if config_id:
            index = int(config_id) - 1
            if index < 0 or index >= len(private_targets):
                raise ValueError(f"配置模型 {config_id} 不存在。")
            merged = dict(private_targets[index])
            for key in ("name", "model", "base_url", "temperature", "timeout", "concurrency", "reasoning_effort"):
                if raw.get(key) not in (None, ""):
                    merged[key] = raw[key]
            resolved.append(merged)
        else:
            resolved.append(raw)
    return resolved


def normalize_targets(raw_targets: list[dict[str, Any]]) -> list[ModelTarget]:
    raw_targets = resolve_config_targets(raw_targets)
    if not raw_targets:
        raise ValueError("至少需要输入 1 个模型接口。")
    if len(raw_targets) > 10:
        raise ValueError("最多支持同时评测 10 个模型接口。")

    targets: list[ModelTarget] = []
    for index, raw in enumerate(raw_targets, start=1):
        base_url = str(raw.get("base_url", "")).strip()
        model = str(raw.get("model", "")).strip()
        api_key = str(raw.get("api_key", "")).strip()
        name = str(raw.get("name", "")).strip() or model or f"模型 {index}"
        if not base_url or not model:
            raise ValueError(f"第 {index} 个接口缺少 base_url 或 model。")
        targets.append(
            ModelTarget(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=float(raw.get("temperature", 0) or 0),
                timeout=int(raw.get("timeout", 360) or 360),
                concurrency=max(1, min(int(raw.get("concurrency", 3) or 3), 8)),
                reasoning_effort=str(raw.get("reasoning_effort", "")).strip(),
            )
        )
    return targets


def normalize_judge_target(raw: dict[str, Any]) -> ModelTarget:
    base_url = str(raw.get("base_url", "")).strip()
    model = str(raw.get("model", "")).strip()
    api_key = str(raw.get("api_key", "")).strip()
    name = str(raw.get("name", "")).strip() or "裁判模型"
    if not base_url or not model:
        raise ValueError("裁判模型缺少 base_url 或 model。")
    return ModelTarget(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0,
        timeout=int(raw.get("timeout", 360) or 360),
        concurrency=max(1, min(int(raw.get("concurrency", 1) or 1), 2)),
        reasoning_effort=str(raw.get("reasoning_effort", "")).strip(),
    )


def call_chat_completion(target: ModelTarget, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
    system_content = system_prompt or (
        "你正在参加一个自动化大模型评测。请严格遵循用户要求，"
        "答案要清晰、准确、简洁。"
    )
    payload = {
        "model": target.model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": target.temperature,
    }
    if target.reasoning_effort:
        payload["reasoning_effort"] = target.reasoning_effort

    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    request = urllib.request.Request(
        target.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=target.timeout) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            data = json.loads(raw_text)
            content = extract_content(data)
            return {
                "ok": True,
                "content": content,
                "latency_ms": elapsed_ms,
                "raw": data,
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "content": "",
            "latency_ms": elapsed_ms,
            "raw": body[:1200],
            "error": f"HTTP {exc.code}: {body[:300]}",
        }
    except Exception as exc:  # noqa: BLE001 - API 网关错误需要完整返回给界面
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "content": "",
            "latency_ms": elapsed_ms,
            "raw": "",
            "error": str(exc),
        }


def extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()

    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict):
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        if parts:
            return "\n".join(parts).strip()

    for key in ("content", "text", "answer"):
        value = data.get(key)
        if isinstance(value, str):
            return value.strip()
    return json.dumps(data, ensure_ascii=False)[:1200]


def score_answer(answer: str, checks: dict[str, Any], latency_ms: int, ok: bool) -> dict[str, Any]:
    if not ok:
        return {
            "score": 0,
            "passed": [],
            "failed": ["接口调用失败"],
            "latency_score": 0,
        }

    passed: list[str] = []
    failed: list[str] = []
    score_parts: list[float] = []
    parsed_json: Any | None = None

    def get_parsed_json() -> Any | None:
        nonlocal parsed_json
        if parsed_json is None:
            parsed_json = parse_json_answer(answer)
        return parsed_json

    if "keywords_all" in checks:
        keywords = [str(item) for item in checks["keywords_all"]]
        hit = sum(1 for item in keywords if contains(answer, item))
        ratio = hit / max(len(keywords), 1)
        score_parts.append(ratio)
        (passed if ratio >= 1 else failed).append(f"必含关键词 {hit}/{len(keywords)}")

    if "keywords_any" in checks:
        keywords = [str(item) for item in checks["keywords_any"]]
        hit = sum(1 for item in keywords if contains(answer, item))
        ratio = min(hit / max(math.ceil(len(keywords) * 0.4), 1), 1)
        score_parts.append(ratio)
        (passed if hit else failed).append(f"可选关键词命中 {hit}/{len(keywords)}")

    if "forbidden_any" in checks:
        forbidden = [str(item) for item in checks["forbidden_any"]]
        hit = [item for item in forbidden if contains(answer, item)]
        ratio = 0 if hit else 1
        score_parts.append(ratio)
        (failed if hit else passed).append(
            f"禁用内容命中：{', '.join(hit)}" if hit else "未命中禁用内容"
        )

    if "numeric_answer" in checks:
        expected = float(checks["numeric_answer"])
        numbers = extract_numbers(answer)
        matched = any(abs(number - expected) < 1e-6 for number in numbers)
        score_parts.append(1 if matched else 0)
        (passed if matched else failed).append(
            f"数值答案 {format_number(expected)} {'匹配' if matched else '未匹配'}"
        )

    if "exact_answer_any" in checks:
        expected_answers = [normalize_text(str(item)) for item in checks["exact_answer_any"]]
        normalized_answer = normalize_text(answer)
        matched = any(expected and expected in normalized_answer for expected in expected_answers)
        score_parts.append(1 if matched else 0)
        (passed if matched else failed).append("精确答案匹配" if matched else "精确答案未匹配")

    if checks.get("must_be_json"):
        parsed = get_parsed_json()
        score_parts.append(1 if parsed is not None else 0)
        (passed if parsed is not None else failed).append("JSON 格式合法" if parsed is not None else "JSON 格式不合法")
        required_keys = checks.get("json_keys_all") or []
        if required_keys:
            if isinstance(parsed, dict):
                hit = sum(1 for key in required_keys if key in parsed)
                ratio = hit / max(len(required_keys), 1)
            else:
                hit = 0
                ratio = 0
            score_parts.append(ratio)
            (passed if ratio >= 1 else failed).append(f"JSON 字段 {hit}/{len(required_keys)}")

    if "json_array_length" in checks:
        expected_length = int(checks["json_array_length"])
        parsed = get_parsed_json()
        actual_length = len(parsed) if isinstance(parsed, list) else None
        matched = actual_length == expected_length
        score_parts.append(1 if matched else 0)
        (passed if matched else failed).append(
            f"JSON 数组长度 {actual_length if actual_length is not None else '非数组'}/{expected_length}"
        )

    if "json_array_length_at" in checks:
        parsed = get_parsed_json()
        for path, expected_length_raw in dict(checks["json_array_length_at"]).items():
            expected_length = int(expected_length_raw)
            value = get_json_path(parsed, str(path))
            actual_length = len(value) if isinstance(value, list) else None
            matched = actual_length == expected_length
            score_parts.append(1 if matched else 0)
            (passed if matched else failed).append(
                f"JSON 数组 {path} 长度 {actual_length if actual_length is not None else '非数组'}/{expected_length}"
            )

    if "json_list_item_keys_exact_at" in checks:
        parsed = get_parsed_json()
        for path, expected_keys_raw in dict(checks["json_list_item_keys_exact_at"]).items():
            expected_keys = {str(item) for item in expected_keys_raw}
            value = get_json_path(parsed, str(path))
            if isinstance(value, list):
                matched_items = [isinstance(item, dict) and set(item.keys()) == expected_keys for item in value]
            else:
                matched_items = []
            ratio = sum(1 for item in matched_items if item) / max(len(matched_items), 1)
            score_parts.append(ratio)
            (passed if ratio >= 1 else failed).append(
                f"JSON 数组 {path} 子字段精确匹配 {sum(1 for item in matched_items if item)}/{len(matched_items) or 1}"
            )

    if "json_keys_exact" in checks:
        expected_keys = {str(item) for item in checks["json_keys_exact"]}
        parsed = get_parsed_json()
        if isinstance(parsed, dict):
            matched_items = [set(parsed.keys()) == expected_keys]
        elif isinstance(parsed, list):
            matched_items = [isinstance(item, dict) and set(item.keys()) == expected_keys for item in parsed]
        else:
            matched_items = []
        ratio = sum(1 for item in matched_items if item) / max(len(matched_items), 1)
        score_parts.append(ratio)
        (passed if ratio >= 1 else failed).append(
            f"JSON 字段精确匹配 {sum(1 for item in matched_items if item)}/{len(matched_items) or 1}"
        )

    if "max_words" in checks:
        max_words = int(checks["max_words"])
        length = len(answer)
        ratio = 1 if length <= max_words else max(0, 1 - (length - max_words) / max(max_words, 1))
        score_parts.append(ratio)
        (passed if ratio >= 1 else failed).append(f"长度 {length}/{max_words}")

    if "line_count" in checks:
        expected_lines = int(checks["line_count"])
        actual_lines = len([line for line in answer.splitlines() if line.strip()])
        ratio = 1 if actual_lines == expected_lines else max(0, 1 - abs(actual_lines - expected_lines) / max(expected_lines, 1))
        score_parts.append(ratio)
        (passed if ratio >= 1 else failed).append(f"行数 {actual_lines}/{expected_lines}")

    if "line_prefix_any" in checks:
        prefixes = tuple(str(item) for item in checks["line_prefix_any"])
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        hit = sum(1 for line in lines if line.startswith(prefixes))
        ratio = hit / max(len(lines), 1) if lines else 0
        score_parts.append(ratio)
        (passed if ratio >= 0.8 else failed).append(f"列表格式 {hit}/{len(lines)}")

    if "regex_any" in checks:
        patterns = [str(item) for item in checks["regex_any"]]
        hit = sum(1 for pattern in patterns if re.search(pattern, answer, flags=re.IGNORECASE))
        ratio = min(hit / max(math.ceil(len(patterns) * 0.4), 1), 1)
        score_parts.append(ratio)
        (passed if hit else failed).append(f"正则命中 {hit}/{len(patterns)}")

    if "regex_all" in checks:
        patterns = [str(item) for item in checks["regex_all"]]
        hit = sum(1 for pattern in patterns if re.search(pattern, answer, flags=re.IGNORECASE | re.MULTILINE))
        ratio = hit / max(len(patterns), 1)
        score_parts.append(ratio)
        (passed if ratio >= 1 else failed).append(f"必含正则 {hit}/{len(patterns)}")

    content_score = statistics.mean(score_parts) if score_parts else 0.5
    latency_score = latency_to_score(latency_ms)
    raw_score = round((content_score * 0.85 + latency_score * 0.15) * 100, 1)
    final_score = stretch_score(raw_score, center=72, factor=1.18)
    return {
        "score": final_score,
        "raw_rule_score": raw_score,
        "passed": passed,
        "failed": failed,
        "latency_score": round(latency_score * 100, 1),
    }


def contains(answer: str, needle: str) -> bool:
    return needle.lower() in answer.lower()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def extract_numbers(text: str) -> list[float]:
    values = []
    for item in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def parse_json_answer(answer: str) -> Any | None:
    text = answer.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None


def get_json_path(value: Any, path: str) -> Any | None:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def latency_to_score(latency_ms: int) -> float:
    if latency_ms <= 1500:
        return 1.0
    if latency_ms <= 5000:
        return 0.85
    if latency_ms <= 12000:
        return 0.65
    if latency_ms <= 30000:
        return 0.4
    return 0.2


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def stretch_score(score: float, center: float = 75, factor: float = 1.15) -> float:
    """轻微拉伸分数分布，让中高分模型更容易拉开差距。"""
    adjusted = center + (float(score) - center) * factor
    return round(max(0, min(adjusted, 100)), 1)


def run_evaluation(
    raw_targets: list[dict[str, Any]],
    case_limit: int | None = None,
    dataset_id: str | None = None,
    judge_enabled: bool = False,
) -> dict[str, Any]:
    selected_dataset_path = dataset_path_from_id(dataset_id)
    dataset = load_dataset(selected_dataset_path)
    cases = dataset["cases"][:case_limit] if case_limit else dataset["cases"]
    targets = normalize_targets(raw_targets)

    report_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    model_results: dict[str, dict[str, Any]] = {
        target.name: {
            "name": target.name,
            "model": target.model,
            "base_url": target.base_url,
            "cases": [],
            "summary": {},
        }
        for target in targets
    }

    future_to_target: dict[Any, str] = {}
    executors = [
        ThreadPoolExecutor(max_workers=target.concurrency, thread_name_prefix=f"eval-{target.name[:12]}")
        for target in targets
    ]
    try:
        for target, executor in zip(targets, executors):
            for case in cases:
                future = executor.submit(evaluate_one_case, target, case)
                future_to_target[future] = target.name
        for future in as_completed(future_to_target):
            item = future.result()
            model_results[item["target_name"]]["cases"].append(item)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    for model in model_results.values():
        model["cases"].sort(key=lambda item: item["case_id"])

    if judge_enabled:
        judge_state = apply_judge_scoring(model_results, cases, load_private_judge())
    else:
        judge_state = {
            "enabled": False,
            "config": None,
            "status": "disabled",
            "note": "本次报告未使用裁判模型细评分。",
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
        }

    for model in model_results.values():
        model["summary"] = summarize_model(model["cases"], cases)

    rankings = build_rankings(list(model_results.values()))
    report = {
        "id": report_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "id": selected_dataset_path.stem,
            "name": dataset["name"],
            "version": dataset["version"],
            "case_count": len(cases),
            "categories": sorted({case["category"] for case in cases}),
        },
        "judge": judge_state,
        "models": list(model_results.values()),
        "rankings": rankings,
        "separation": build_separation_metrics(rankings),
        "insights": build_insights(rankings, list(model_results.values())),
    }
    save_report(report)
    return report


def evaluate_one_case(target: ModelTarget, case: dict[str, Any]) -> dict[str, Any]:
    call = call_chat_completion(target, case["prompt"])
    scored = score_answer(call["content"], case.get("checks", {}), call["latency_ms"], call["ok"])
    return {
        "target_name": target.name,
        "case_id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "prompt": case["prompt"],
        "answer": call["content"],
        "ok": call["ok"],
        "error": call["error"],
        "latency_ms": call["latency_ms"],
        "score": scored["score"],
        "raw_rule_score": scored["raw_rule_score"],
        "latency_score": scored["latency_score"],
        "passed": scored["passed"],
        "failed": scored["failed"],
        "weight": float(case.get("weight", 1.0)),
    }


def apply_judge_scoring(
    model_results: dict[str, dict[str, Any]],
    cases: list[dict[str, Any]],
    judge_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not judge_config:
        return {
            "enabled": True,
            "config": None,
            "status": "missing_config",
            "note": "已勾选裁判评分，但未找到可用裁判模型配置。",
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
        }

    try:
        judge_target = normalize_judge_target(judge_config)
    except Exception as exc:  # noqa: BLE001 - 配置错误需要写入报告
        return {
            "enabled": True,
            "config": public_private_judge(),
            "status": "config_error",
            "note": f"裁判模型配置不可用：{exc}",
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
        }

    case_map = {case["id"]: case for case in cases}
    attempted = 0
    succeeded = 0
    failed = 0
    futures: dict[Any, dict[str, Any]] = {}
    max_workers = max(1, min(judge_target.concurrency, 2))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="judge") as executor:
        for model in model_results.values():
            for item in model["cases"]:
                item["rule_score"] = item["score"]
                if not item.get("ok"):
                    item["judge"] = {
                        "ok": False,
                        "error": "模型接口调用失败，裁判不再评分。",
                    }
                    continue
                case = case_map.get(item["case_id"])
                if not case:
                    continue
                attempted += 1
                futures[executor.submit(judge_one_answer, judge_target, case, item)] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                judge_result = future.result()
            except Exception as exc:  # noqa: BLE001 - 单条裁判失败不应中断整份报告
                judge_result = {"ok": False, "error": str(exc)}
            item["judge"] = judge_result
            if judge_result.get("ok"):
                succeeded += 1
                item["judge_score"] = judge_result["score"]
                item["score"] = merge_rule_and_judge_score(float(item["rule_score"]), float(judge_result["score"]))
                reason = str(judge_result.get("reason") or "").strip()
                if reason:
                    item.setdefault("failed", []).append(f"裁判评语：{reason[:160]}")
            else:
                failed += 1
                item["judge_score"] = None

    return {
        "enabled": True,
        "config": public_private_judge(),
        "status": "scored" if succeeded else "failed",
        "note": f"裁判评分完成：成功 {succeeded} 条，失败 {failed} 条。最终分 = 规则分 40% + 裁判分 60%，并进行区分度拉伸。",
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
    }


def judge_one_answer(judge_target: ModelTarget, case: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    prompt = build_judge_prompt(case, item)
    system_prompt = (
        "你是大模型评测裁判。请只根据给定题目、评分规则和参赛模型回答打分，"
        "不要因为文风好就给高分。必须输出一个 JSON 对象，不要输出 Markdown。"
    )
    call = call_chat_completion(judge_target, prompt, system_prompt=system_prompt)
    if not call["ok"]:
        return {
            "ok": False,
            "error": call["error"],
            "latency_ms": call["latency_ms"],
        }
    parsed = parse_json_answer(call["content"])
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "error": "裁判未返回合法 JSON。",
            "latency_ms": call["latency_ms"],
            "raw": call["content"][:800],
        }
    score = clamp_score(parsed.get("score"))
    if score is None:
        return {
            "ok": False,
            "error": "裁判 JSON 缺少 0-100 的 score。",
            "latency_ms": call["latency_ms"],
            "raw": parsed,
        }
    subscores = parsed.get("subscores")
    if not isinstance(subscores, dict):
        subscores = {}
    errors = parsed.get("errors")
    if not isinstance(errors, list):
        errors = []
    return {
        "ok": True,
        "score": score,
        "reason": str(parsed.get("reason") or "")[:500],
        "subscores": {str(key): clamp_score(value) for key, value in subscores.items() if clamp_score(value) is not None},
        "errors": [str(error)[:120] for error in errors[:5]],
        "confidence": clamp_float(parsed.get("confidence"), 0, 1),
        "latency_ms": call["latency_ms"],
    }


def build_judge_prompt(case: dict[str, Any], item: dict[str, Any]) -> str:
    checks = json.dumps(case.get("checks", {}), ensure_ascii=False, indent=2)
    failed = "；".join(str(x) for x in item.get("failed", [])) or "无"
    passed = "；".join(str(x) for x in item.get("passed", [])) or "无"
    return f"""请评估下面这个参赛模型回答。评分要严格，分数要拉开：

题目 ID：{case.get("id")}
能力维度：{case.get("category")}
难度：{case.get("difficulty")}
题目：
{case.get("prompt")}

规则检查项：
{checks}

规则初评分：{item.get("rule_score", item.get("score"))}
规则通过项：{passed}
规则扣分项：{failed}

参赛模型回答：
{item.get("answer") or "(无回答)"}

请按 0-100 分给出裁判分。参考 rubric：
- correctness：事实、计算、结论是否正确，40 分
- reasoning：推理链、约束处理、证据质量，25 分
- instruction_following：是否严格满足格式、边界、角色和输出要求，20 分
- robustness：抗干扰、安全边界、异常处理，10 分
- clarity：表达清晰度，5 分

评分要求：
- 明显跑题、遗漏关键约束或输出格式错误，应低于 60。
- 只答对表面但没有处理隐含约束，应在 60-75。
- 基本正确但有小缺陷，应在 75-88。
- 只有完整、严格、可执行且无关键漏洞，才可高于 90。
- 不要给人情分，模型之间需要有区分度。

只输出 JSON：
{{
  "score": 0,
  "subscores": {{
    "correctness": 0,
    "reasoning": 0,
    "instruction_following": 0,
    "robustness": 0,
    "clarity": 0
  }},
  "reason": "一句话说明主要给分原因",
  "errors": ["最多列出3个主要问题"],
  "confidence": 0.0
}}"""


def merge_rule_and_judge_score(rule_score: float, judge_score: float) -> float:
    mixed = rule_score * 0.4 + judge_score * 0.6
    return stretch_score(mixed, center=78, factor=1.25)


def clamp_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0, min(score, 100)), 1)


def clamp_float(value: Any, min_value: float, max_value: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(min_value, min(number, max_value)), 3)


def summarize_model(results: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    weighted_total = sum(item["score"] * item["weight"] for item in results)
    weight_sum = sum(item["weight"] for item in results) or 1
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_category.setdefault(item["category"], []).append(item)

    category_scores = {
        category: round(
            sum(item["score"] * item["weight"] for item in values)
            / (sum(item["weight"] for item in values) or 1),
            1,
        )
        for category, values in sorted(by_category.items())
    }
    latencies = [item["latency_ms"] for item in results if item["ok"]]
    pass_count = sum(1 for item in results if item["score"] >= 70)
    total = len(cases)
    overall = round(weighted_total / weight_sum, 1)
    return {
        "overall_score": overall,
        "grade": grade_model(overall),
        "pass_rate": round(pass_count / max(total, 1) * 100, 1),
        "avg_latency_ms": round(statistics.mean(latencies)) if latencies else None,
        "success_count": sum(1 for item in results if item["ok"]),
        "case_count": total,
        "category_scores": category_scores,
        "strengths": top_categories(category_scores, reverse=True),
        "weaknesses": top_categories(category_scores, reverse=False),
    }


def grade_model(score: float) -> str:
    if score >= 90:
        return "S 级：综合旗舰"
    if score >= 80:
        return "A 级：生产可用"
    if score >= 70:
        return "B 级：可用于轻量任务"
    if score >= 60:
        return "C 级：需要场景限制"
    return "D 级：暂不建议上线"


def top_categories(scores: dict[str, float], reverse: bool) -> list[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=reverse)[:3]]


def build_rankings(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(models, key=lambda item: item["summary"]["overall_score"], reverse=True)
    return [
        {
            "rank": index,
            "name": model["name"],
            "model": model["model"],
            "overall_score": model["summary"]["overall_score"],
            "grade": model["summary"]["grade"],
            "pass_rate": model["summary"]["pass_rate"],
            "avg_latency_ms": model["summary"]["avg_latency_ms"],
        }
        for index, model in enumerate(ranked, start=1)
    ]


def build_separation_metrics(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["overall_score"] for row in rankings]
    if len(scores) < 2:
        return {
            "score_range": 0,
            "top_gap": 0,
            "top4_range": 0,
            "stddev": 0,
            "level": "样本不足",
            "note": "至少需要 2 个模型才能判断区分度。",
        }
    score_range = round(max(scores) - min(scores), 1)
    top_gap = round(scores[0] - scores[1], 1)
    top4_scores = scores[: min(4, len(scores))]
    top4_range = round(max(top4_scores) - min(top4_scores), 1)
    stddev = round(statistics.pstdev(scores), 1)
    if (score_range >= 15 or stddev >= 6) and top_gap >= 3:
        level = "区分度较好"
    elif score_range >= 8 or stddev >= 3 or top4_range >= 6:
        level = "区分度一般"
    else:
        level = "区分度不足"
    if top4_range < 5 and len(scores) >= 4:
        note = "整体分差可能由尾部模型拉大，但头部模型仍接近；建议使用诊断题和复测拉开关键能力差异。"
    else:
        note = "分差越小，越需要增加高难题、稳定性复测或裁判模型评分。"
    return {
        "score_range": score_range,
        "top_gap": top_gap,
        "top4_range": top4_range,
        "stddev": stddev,
        "level": level,
        "note": note,
    }


def build_insights(rankings: list[dict[str, Any]], models: list[dict[str, Any]]) -> list[str]:
    if not rankings:
        return []
    best = rankings[0]
    insights = [
        f"综合排名第一：{best['name']}，总分 {best['overall_score']}，分类为 {best['grade']}。",
    ]
    separation = build_separation_metrics(rankings)
    insights.append(
        f"排名区分度：{separation['level']}，最高最低分差 {separation['score_range']}，第一第二分差 {separation['top_gap']}。"
    )
    fastest = min(
        (model for model in models if model["summary"]["avg_latency_ms"] is not None),
        key=lambda item: item["summary"]["avg_latency_ms"],
        default=None,
    )
    if fastest:
        insights.append(
            f"平均响应最快：{fastest['name']}，平均延迟 {fastest['summary']['avg_latency_ms']} ms。"
        )
    for model in models:
        weaknesses = "、".join(model["summary"].get("weaknesses", [])[:2])
        if weaknesses:
            insights.append(f"{model['name']} 后续重点观察：{weaknesses}。")
    return insights


def save_report(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{report['id']}.json"
    md_path = REPORT_DIR / f"{report['id']}.md"
    html_path = REPORT_DIR / f"{report['id']}.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    html_path.write_text(render_html_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# 大模型自动评测报告 {report['id']}",
        "",
        f"- 生成时间：{report['created_at']}",
        f"- 测试集：{report['dataset']['name']} v{report['dataset']['version']}",
        f"- 题目数量：{report['dataset']['case_count']}",
        "",
        "## 综合排名",
        "",
        "| 排名 | 模型名称 | 模型 ID | 总分 | 分类 | 通过率 | 平均延迟 |",
        "|---:|---|---|---:|---|---:|---:|",
    ]
    for row in report["rankings"]:
        latency = "-" if row["avg_latency_ms"] is None else f"{row['avg_latency_ms']} ms"
        lines.append(
            f"| {row['rank']} | {row['name']} | {row['model']} | {row['overall_score']} | "
            f"{row['grade']} | {row['pass_rate']}% | {latency} |"
        )

    lines.extend(["", "## 关键结论", ""])
    lines.extend(f"- {item}" for item in report["insights"])
    if report.get("separation"):
        separation = report["separation"]
        lines.extend(
            [
                "",
                "## 区分度诊断",
                "",
                f"- 等级：{separation['level']}",
                f"- 最高最低分差：{separation['score_range']}",
                f"- 第一第二分差：{separation['top_gap']}",
                f"- 前四名分差：{separation.get('top4_range', '-')}",
                f"- 分数标准差：{separation['stddev']}",
                f"- 说明：{separation['note']}",
            ]
        )

    for model in report["models"]:
        summary = model["summary"]
        lines.extend(
            [
                "",
                f"## {model['name']}",
                "",
                f"- 总分：{summary['overall_score']}",
                f"- 分类：{summary['grade']}",
                f"- 通过率：{summary['pass_rate']}%",
                f"- 平均延迟：{summary['avg_latency_ms'] or '-'} ms",
                f"- 优势维度：{'、'.join(summary['strengths'])}",
                f"- 待加强维度：{'、'.join(summary['weaknesses'])}",
                "",
                "| 题目 | 维度 | 难度 | 最终分 | 规则分 | 裁判分 | 延迟 | 主要扣分点 |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in model["cases"]:
            failed = "；".join(item["failed"][:3]) if item["failed"] else "-"
            rule_score = item.get("rule_score", item["score"])
            judge_score = item.get("judge_score")
            judge_text = "-" if judge_score is None else str(judge_score)
            lines.append(
                f"| {item['case_id']} | {item['category']} | {item['difficulty']} | "
                f"{item['score']} | {rule_score} | {judge_text} | {item['latency_ms']} ms | {failed} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_html_report(report: dict[str, Any]) -> str:
    best = report["rankings"][0] if report["rankings"] else {}
    generated = escape(str(report["created_at"]))
    dataset = report["dataset"]
    ranking_rows = "\n".join(render_html_ranking_row(row) for row in report["rankings"])
    insight_items = "\n".join(f"<li>{escape(item)}</li>" for item in report["insights"])
    model_sections = "\n".join(render_html_model_section(model) for model in report["models"])
    separation = report.get("separation") or {}
    separation_html = render_html_separation(separation)
    judge_html = render_html_judge_status(report.get("judge") or {})
    title = f"大模型自动评测报告 {escape(report['id'])}"
    best_name = escape(str(best.get("name", "-")))
    best_score = escape(str(best.get("overall_score", "-")))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --paper: #ffffff;
      --ink: #17202a;
      --muted: #637083;
      --line: #dce3eb;
      --primary: #176b5f;
      --primary-2: #e6f4f1;
      --accent: #d97706;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      line-height: 1.55;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 32px; }}
    .hero {{
      background: #111820;
      color: #fff;
      border-radius: 8px;
      padding: 34px;
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(260px, .6fr);
      gap: 24px;
      align-items: end;
    }}
    .eyebrow {{ color: #83d8cc; font-size: 13px; font-weight: 800; margin: 0 0 8px; }}
    h1 {{ margin: 0; font-size: 34px; line-height: 1.2; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 0; font-size: 18px; }}
    .hero p {{ color: #c6d1dc; margin: 14px 0 0; }}
    .hero-card {{
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.13);
      border-radius: 8px;
      padding: 18px;
    }}
    .hero-card span {{ color: #b9c8d5; font-size: 13px; }}
    .hero-card strong {{ display: block; font-size: 34px; margin-top: 4px; }}
    .grid {{ display: grid; gap: 18px; }}
    .summary-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 20px 0; }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 14px 45px rgba(24, 39, 54, .07);
    }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; font-size: 26px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 13px; }}
    .score {{
      display: inline-flex;
      min-width: 58px;
      justify-content: center;
      padding: 5px 10px;
      border-radius: 999px;
      background: var(--primary-2);
      color: var(--primary);
      font-weight: 800;
    }}
    .rank-1 {{ color: var(--accent); font-weight: 800; }}
    .insights {{ margin: 0; padding-left: 20px; }}
    .insights li {{ margin: 8px 0; }}
    .model-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }}
    .model-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 14px; }}
    .model-head p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .bars {{ display: grid; gap: 10px; margin-top: 14px; }}
    .bar-row {{ display: grid; grid-template-columns: 96px minmax(0, 1fr) 48px; gap: 10px; align-items: center; font-size: 13px; }}
    .bar-link {{ color: inherit; text-decoration: none; border-radius: 6px; padding: 3px 4px; margin: -3px -4px; }}
    .bar-link:hover {{ background: #eef5f3; }}
    .bar {{ height: 10px; background: #e8edf3; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: var(--primary); }}
    .weak-list {{ margin-top: 14px; display: grid; gap: 8px; }}
    .weak-item {{ background: #fbfcfd; border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .weak-link {{ color: inherit; text-decoration: none; display: block; }}
    .weak-link:hover {{ border-color: var(--primary); background: #f3faf8; }}
    .weak-item strong {{ display: flex; justify-content: space-between; gap: 10px; }}
    .weak-item p {{ margin: 6px 0 0; color: var(--muted); font-size: 13px; }}
    .score-details {{ margin-top: 16px; border-top: 1px solid var(--line); padding-top: 12px; }}
    .score-details > summary {{ cursor: pointer; font-weight: 800; color: var(--primary); }}
    .detail-category {{ scroll-margin-top: 18px; margin-top: 16px; border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fcfdfd; }}
    .detail-category h4 {{ display: flex; justify-content: space-between; gap: 10px; margin: 0 0 10px; font-size: 15px; }}
    .detail-category h4 span {{ color: var(--primary); }}
    .case-detail-list {{ display: grid; gap: 10px; }}
    .case-detail {{ scroll-margin-top: 18px; border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }}
    .case-detail.api-error {{ border-color: #f2b8b5; background: #fff8f7; }}
    .case-detail.low-score {{ border-color: #f4d19b; background: #fffaf2; }}
    .case-detail-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    .case-meta, .error-line {{ color: var(--muted); margin: 8px 0; font-size: 13px; }}
    .error-line {{ color: var(--danger); }}
    .score-breakdown {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    .score-breakdown span {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 9px; color: var(--muted); font-size: 12px; background: #fbfcfd; }}
    .judge-box {{ margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #f8fbfa; }}
    .judge-box strong {{ color: var(--primary); }}
    .judge-box p {{ margin: 6px 0 0; color: var(--muted); font-size: 13px; }}
    .judge-sub {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .judge-sub span {{ border-radius: 999px; background: #eaf5f2; color: var(--primary); padding: 4px 8px; font-size: 12px; }}
    .case-detail details {{ margin-top: 8px; }}
    .case-detail summary {{ cursor: pointer; color: var(--primary); font-weight: 700; }}
    .case-detail pre {{ white-space: pre-wrap; word-break: break-word; margin: 8px 0 0; padding: 10px; background: #f3f6f8; border-radius: 6px; font-family: Consolas, "Microsoft YaHei", monospace; font-size: 12px; }}
    .check-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }}
    .check-grid div {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfd; }}
    .check-grid p {{ margin: 6px 0 0; color: var(--muted); font-size: 13px; }}
    .footer {{ color: var(--muted); text-align: center; font-size: 12px; margin: 28px 0 4px; }}
    @media (max-width: 900px) {{
      .page {{ padding: 18px; }}
      .hero, .summary-grid, .model-grid {{ grid-template-columns: 1fr; }}
      .check-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
    }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ max-width: none; padding: 0; }}
      .card, .hero {{ box-shadow: none; break-inside: avoid; }}
      a {{ color: inherit; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <p class="eyebrow">LLM Evaluation Report</p>
        <h1>大模型自动评测报告</h1>
        <p>测试集：{escape(dataset["name"])} v{escape(dataset["version"])} · 生成时间：{generated}</p>
      </div>
      <div class="hero-card">
        <span>综合第一</span>
        <strong>{best_name}</strong>
        <span>总分 {best_score}</span>
      </div>
    </section>

    <section class="grid summary-grid">
      <div class="card metric"><span>参评模型</span><strong>{len(report["models"])}</strong></div>
      <div class="card metric"><span>测试题数</span><strong>{dataset["case_count"]}</strong></div>
      <div class="card metric"><span>能力维度</span><strong>{len(dataset["categories"])}</strong></div>
      <div class="card metric"><span>报告 ID</span><strong>{escape(report["id"][-8:])}</strong></div>
    </section>

    {judge_html}

    <section class="card">
      <h2>综合排名</h2>
      <table>
        <thead>
          <tr><th>排名</th><th>模型</th><th>总分</th><th>分类</th><th>通过率</th><th>平均延迟</th></tr>
        </thead>
        <tbody>{ranking_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>关键结论</h2>
      <ul class="insights">{insight_items}</ul>
    </section>

    {separation_html}

    <section class="grid model-grid">
      {model_sections}
    </section>

    <p class="footer">由 LLM Auto Evaluator 生成。建议结合业务测试集、成本和人工复核做最终模型选型。</p>
  </main>
</body>
</html>"""


def render_html_ranking_row(row: dict[str, Any]) -> str:
    rank_class = " class=\"rank-1\"" if row["rank"] == 1 else ""
    latency = "-" if row["avg_latency_ms"] is None else f"{row['avg_latency_ms']} ms"
    return (
        f"<tr><td{rank_class}>#{row['rank']}</td>"
        f"<td><strong>{escape(str(row['name']))}</strong><br><span>{escape(str(row['model']))}</span></td>"
        f"<td><span class=\"score\">{escape(str(row['overall_score']))}</span></td>"
        f"<td>{escape(str(row['grade']))}</td>"
        f"<td>{escape(str(row['pass_rate']))}%</td>"
        f"<td>{escape(latency)}</td></tr>"
    )


def render_html_model_section(model: dict[str, Any]) -> str:
    summary = model["summary"]
    model_anchor = slugify(model["name"])
    bars = "\n".join(
        f"""<a class="bar-row bar-link" href="#detail-{model_anchor}-{slugify(category)}"><span>{escape(category)}</span><div class="bar"><span style="width:{max(3, score)}%"></span></div><strong>{score}</strong></a>"""
        for category, score in summary["category_scores"].items()
    )
    weak_cases = [item for item in model["cases"] if item["score"] < 85 or item["error"]][:4]
    if weak_cases:
        weak_html = "\n".join(render_html_weak_case(item) for item in weak_cases)
    else:
        weak_html = "<div class=\"weak-item\"><strong>未发现待加强题目</strong><p>该模型在当前测试集上表现稳定。</p></div>"
    latency = "-" if summary["avg_latency_ms"] is None else f"{summary['avg_latency_ms']} ms"
    detail_sections = "\n".join(render_html_category_detail(model, category) for category in summary["category_scores"])
    return f"""<article class="card" id="model-{model_anchor}">
      <div class="model-head">
        <div>
          <h3>{escape(str(model["name"]))}</h3>
          <p>{escape(str(model["model"]))} · {summary["success_count"]}/{summary["case_count"]} 调用成功</p>
        </div>
        <span class="score">{summary["overall_score"]}</span>
      </div>
      <table>
        <tr><th>分类</th><td>{escape(str(summary["grade"]))}</td></tr>
        <tr><th>通过率</th><td>{summary["pass_rate"]}%</td></tr>
        <tr><th>平均延迟</th><td>{latency}</td></tr>
        <tr><th>优势维度</th><td>{escape("、".join(summary["strengths"]))}</td></tr>
        <tr><th>待加强维度</th><td>{escape("、".join(summary["weaknesses"]))}</td></tr>
      </table>
      <div class="bars">{bars}</div>
      <div class="weak-list">{weak_html}</div>
      <details class="score-details" open>
        <summary>评分过程明细</summary>
        {detail_sections}
      </details>
    </article>"""


def render_html_separation(separation: dict[str, Any]) -> str:
    if not separation:
        return ""
    return f"""<section class="card">
      <h2>区分度诊断</h2>
      <section class="grid summary-grid" style="margin:0">
        <div class="metric"><span>等级</span><strong>{escape(str(separation["level"]))}</strong></div>
        <div class="metric"><span>最高最低分差</span><strong>{escape(str(separation["score_range"]))}</strong></div>
        <div class="metric"><span>第一第二分差</span><strong>{escape(str(separation["top_gap"]))}</strong></div>
        <div class="metric"><span>前四名分差</span><strong>{escape(str(separation.get("top4_range", "-")))}</strong></div>
      </section>
      <p style="color:var(--muted);margin:14px 0 0">{escape(str(separation["note"]))}</p>
    </section>"""


def render_html_judge_status(judge: dict[str, Any]) -> str:
    if not judge:
        return ""
    enabled = bool(judge.get("enabled"))
    config = judge.get("config") or {}
    status_map = {
        "scored": "已评分",
        "failed": "评分失败",
        "missing_config": "缺少配置",
        "config_error": "配置错误",
        "disabled": "未启用",
    }
    status = status_map.get(str(judge.get("status") or ""), "已启用" if enabled else "未启用")
    model = config.get("model") or "-"
    name = config.get("name") or "-"
    note = judge.get("note") or ("本次报告未使用裁判模型细评分。" if not enabled else "")
    return f"""<section class="card">
      <h2>裁判评分</h2>
      <section class="grid summary-grid" style="margin:0">
        <div class="metric"><span>状态</span><strong>{escape(status)}</strong></div>
        <div class="metric"><span>裁判模型</span><strong>{escape(str(model))}</strong></div>
        <div class="metric"><span>成功/尝试</span><strong>{escape(str(judge.get("succeeded", 0)))}/{escape(str(judge.get("attempted", 0)))}</strong></div>
        <div class="metric"><span>模式</span><strong>Rubric 混合</strong></div>
      </section>
      <p style="color:var(--muted);margin:14px 0 0">{escape(str(name))} · {escape(str(note))}</p>
    </section>"""


def render_html_weak_case(item: dict[str, Any]) -> str:
    failed = "；".join(item["failed"][:2]) if item["failed"] else "无明显扣分点"
    if item["error"]:
        failed = item["error"]
    return (
        f"<a class=\"weak-item weak-link\" href=\"#{case_anchor(item)}\"><strong><span>{escape(str(item['case_id']))} · "
        f"{escape(str(item['category']))}</span><span>{item['score']}</span></strong>"
        f"<p>{escape(failed)}</p></a>"
    )


def render_html_category_detail(model: dict[str, Any], category: str) -> str:
    model_anchor = slugify(model["name"])
    cases = [item for item in model["cases"] if item["category"] == category]
    rows = "\n".join(render_html_case_detail(item) for item in cases)
    avg = model["summary"]["category_scores"].get(category, "-")
    return f"""<section class="detail-category" id="detail-{model_anchor}-{slugify(category)}">
        <h4>{escape(category)} <span>{escape(str(avg))}</span></h4>
        <div class="case-detail-list">{rows}</div>
      </section>"""


def render_html_case_detail(item: dict[str, Any]) -> str:
    passed = "；".join(str(x) for x in item.get("passed", [])) or "-"
    failed = "；".join(str(x) for x in item.get("failed", [])) or "-"
    answer = str(item.get("answer") or "").strip() or "(无回答)"
    prompt = str(item.get("prompt") or "")
    error = str(item.get("error") or "")
    status_class = "api-error" if error else ("low-score" if float(item.get("score", 0)) < 70 else "ok-score")
    error_html = f"<p class=\"error-line\"><strong>接口错误：</strong>{escape(error)}</p>" if error else ""
    rule_score = item.get("rule_score", item.get("score"))
    judge_score = item.get("judge_score")
    judge_text = "-" if judge_score is None else str(judge_score)
    judge = item.get("judge") if isinstance(item.get("judge"), dict) else {}
    judge_reason = str(judge.get("reason") or judge.get("error") or "").strip()
    judge_subscores = judge.get("subscores") if isinstance(judge.get("subscores"), dict) else {}
    judge_sub_html = "".join(
        f"<span>{escape(str(key))}: {escape(str(value))}</span>" for key, value in judge_subscores.items()
    )
    judge_html = ""
    if judge:
        judge_html = f"""<div class="judge-box">
            <strong>裁判评分</strong>
            <p>裁判分：{escape(judge_text)}；最终分 = 规则分 40% + 裁判分 60%，并进行区分度拉伸。</p>
            {"<p>评语：" + escape(judge_reason) + "</p>" if judge_reason else ""}
            {f"<div class=\"judge-sub\">{judge_sub_html}</div>" if judge_sub_html else ""}
          </div>"""
    return f"""<article class="case-detail {status_class}" id="{case_anchor(item)}">
          <div class="case-detail-head">
            <strong>{escape(str(item["case_id"]))}</strong>
            <span class="score">{escape(str(item["score"]))}</span>
          </div>
          <p class="case-meta">难度：{escape(str(item["difficulty"]))} · 延迟：{escape(str(item["latency_ms"]))} ms · 调用：{"成功" if item.get("ok") else "失败"}</p>
          <div class="score-breakdown">
            <span>规则分：{escape(str(rule_score))}</span>
            <span>裁判分：{escape(judge_text)}</span>
            <span>最终分：{escape(str(item["score"]))}</span>
          </div>
          {error_html}
          {judge_html}
          <details>
            <summary>题目</summary>
            <pre>{escape(prompt)}</pre>
          </details>
          <details>
            <summary>模型回答</summary>
            <pre>{escape(answer)}</pre>
          </details>
          <div class="check-grid">
            <div><strong>通过项</strong><p>{escape(passed)}</p></div>
            <div><strong>扣分项</strong><p>{escape(failed)}</p></div>
          </div>
        </article>"""


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", str(value)).strip("-")
    return text or "item"


def case_anchor(item: dict[str, Any]) -> str:
    return f"case-{slugify(str(item.get('target_name', 'model')))}-{slugify(str(item.get('case_id', 'case')))}"
