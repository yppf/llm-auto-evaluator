from __future__ import annotations

import json
import mimetypes
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from evaluator import (
    DATASET_PATH,
    REPORT_DIR,
    load_dataset,
    list_datasets,
    public_private_judge,
    public_private_targets,
    render_markdown_report,
    render_html_report,
    run_evaluation,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
HOST = "127.0.0.1"
PORT = 8765
APP_VERSION = "judge-scoring-v1"

JOBS: dict[str, dict[str, Any]] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "LLMAutoEvaluator/0.2"

    def do_GET(self) -> None:  # noqa: N802 - http.server 固定命名
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/dataset":
                self.send_json(load_dataset(DATASET_PATH))
                return
            if path == "/api/datasets":
                self.send_json({"datasets": list_datasets()})
                return
            if path == "/api/config-targets":
                self.send_json({"targets": public_private_targets()})
                return
            if path == "/api/judge-config":
                self.send_json({"judge": public_private_judge()})
                return
            if path == "/api/version":
                self.send_json(
                    {
                        "version": APP_VERSION,
                        "server_file": str(Path(__file__).resolve()),
                        "root": str(ROOT),
                        "now": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                return
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/").strip("/")
                self.send_json(JOBS.get(job_id) or {"status": "not_found", "error": "任务不存在"})
                return
            if path.startswith("/api/reports/"):
                self.handle_report(path)
                return
            self.serve_static(path)
        except Exception as exc:  # noqa: BLE001 - 顶层 Web 错误需要返回给前端
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802 - http.server 固定命名
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/evaluations":
                body = self.read_json()
                job_id = body.get("job_id") or f"job-{len(JOBS) + 1}"
                targets = body.get("targets") or []
                case_limit = body.get("case_limit")
                dataset_id = body.get("dataset_id")
                judge_enabled = bool(body.get("judge_enabled"))
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "running",
                    "progress": "评测任务已启动，正在并发请求模型接口。",
                    "result": None,
                    "error": "",
                }
                EXECUTOR.submit(run_job, job_id, targets, case_limit, dataset_id, judge_enabled)
                self.send_json({"job_id": job_id, "status": "running"})
                return
            self.send_json({"error": "接口不存在"}, status=404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=500)

    def serve_static(self, path: str) -> None:
        if path in ("", "/"):
            file_path = STATIC_DIR / "index.html"
        else:
            requested = path.lstrip("/")
            file_path = (STATIC_DIR / requested).resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self.send_error(403)
                return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_report(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            self.send_json({"error": "缺少报告 ID"}, status=400)
            return
        report_id = parts[2]
        fmt = parts[3] if len(parts) > 3 else "json"
        if fmt == "markdown":
            json_path = REPORT_DIR / f"{report_id}.json"
            if not json_path.exists():
                self.send_json({"error": "报告不存在"}, status=404)
                return
            report = json.loads(json_path.read_text(encoding="utf-8"))
            content = render_markdown_report(report).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename={report_id}.md")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if fmt == "html":
            json_path = REPORT_DIR / f"{report_id}.json"
            if not json_path.exists():
                self.send_json({"error": "报告不存在"}, status=404)
                return
            report = json.loads(json_path.read_text(encoding="utf-8"))
            content = render_html_report(report).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Template-Version", APP_VERSION)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        json_path = REPORT_DIR / f"{report_id}.json"
        if not json_path.exists():
            self.send_json({"error": "报告不存在"}, status=404)
            return
        self.send_json(json.loads(json_path.read_text(encoding="utf-8")))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def run_job(
    job_id: str,
    targets: list[dict[str, Any]],
    case_limit: int | None,
    dataset_id: str | None,
    judge_enabled: bool = False,
) -> None:
    try:
        report = run_evaluation(
            targets,
            case_limit=case_limit,
            dataset_id=dataset_id,
            judge_enabled=judge_enabled,
        )
        JOBS[job_id].update(
            {
                "status": "completed",
                "progress": "评测完成，报告已生成。",
                "result": report,
                "report_id": report["id"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        JOBS[job_id].update(
            {
                "status": "failed",
                "progress": "评测失败。",
                "error": str(exc),
            }
        )


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"LLM 自动评测工具已启动：http://{HOST}:{PORT}")
    print("按 Ctrl+C 停止服务。")
    server.serve_forever()


if __name__ == "__main__":
    main()
