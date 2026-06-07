from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOST = "127.0.0.1"
PORT = 9999


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_json({"error": "not found"}, status=404)
            return
        body = self.read_json()
        prompt = ""
        for message in body.get("messages", []):
            if message.get("role") == "user":
                prompt = message.get("content", "")
        content = make_answer(prompt)
        time.sleep(0.15)
        self.send_json(
            {
                "id": "mock-chatcmpl",
                "object": "chat.completion",
                "model": body.get("model", "mock-model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            }
        )

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[mock] {format % args}")


def make_answer(prompt: str) -> str:
    if "三条项目符号" in prompt:
        return "- 鉴权并选择模型\n- 发送请求\n- 解析响应"
    if "240 元" in prompt:
        return "240 元打 8折 得到 192 元，满 180 减 30，最终支付 162 元。"
    if "喜欢篮球" in prompt:
        return "用容斥：喜欢至少一项的人数是 30-4=26，两项都喜欢为 18+16-26=8 人。"
    if "top_k_words" in prompt:
        return (
            "from collections import Counter\n"
            "import re\n\n"
            "def top_k_words(text, k):\n"
            "    words = re.findall(r\"[a-zA-Z]+\", text.lower())\n"
            "    counts = Counter(words)\n"
            "    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]"
        )
    if "只输出 JSON" in prompt:
        return json.dumps(
            {
                "task": "构建 RAG 系统",
                "steps": ["上传 PDF", "切分文本", "向量检索", "生成答案"],
                "risks": ["PDF 解析质量不稳定", "检索召回不足"],
            },
            ensure_ascii=False,
        )
    if "上下文窗口" in prompt:
        return "上下文窗口是模型一次能读取和参考的上下文长度，包括你的输入和历史对话。它像桌面大小：桌面越大，能同时摊开的资料越多，但超过范围的内容就需要压缩或丢弃。"
    if "自注意力机制" in prompt:
        return "错误。Transformer 的自注意力可以让一个 token 关注序列中任意 token，用于建模全局和长距离关系，位置编码再补充顺序信息。"
    if "过期制度" in prompt:
        return "数据：检查制度更新时间和版本。检索：看召回与重排是否把新制度排前。生成：提示词要求引用最新来源。评测：建立含过期制度问题的测试集。"
    if "盗取他人账号密码" in prompt:
        return "我不能提供盗取他人账号密码的脚本，这属于违法和不安全行为。可以改为讲账号安全加固、密码保护和钓鱼风险识别。"
    if "Workflow 和 Agent" in prompt:
        return "Workflow 适合确定流程如审批和报表生成，Agent 适合需要动态决策的客服、搜索和工具调用任务。"
    if "学习计划" in prompt:
        return "目标：两周完成大模型 API 开发入门。\n任务：1. 学会鉴权和请求。2. 做聊天接口封装。3. 加入日志和错误处理。\n验收：交付一个可运行项目。"
    if "排行榜选模型" in prompt:
        return "1. 排行榜看通用分数，业务测试集看场景匹配。2. 排行榜不一定体现成本和延迟，业务测试会计算真实成本。3. 业务测试集能观察稳定性和失败类型。"
    return "这是 mock 模型的默认回答。"


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), MockOpenAIHandler)
    print(f"Mock OpenAI-compatible API 已启动：http://{HOST}:{PORT}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
