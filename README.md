# LLM Auto Evaluator

一个面向 OpenAI-compatible API 的大模型自动评测平台。它支持同时评测多个模型接口，内置中文综合能力测试集，提供规则评分、裁判模型评分、能力维度分析、排名和 HTML/Markdown/JSON 报告导出。

## 功能特性

- 最多同时评测 10 个模型接口。
- 支持 OpenAI-compatible `POST /v1/chat/completions`。
- 内置多套中文测试集，覆盖指令遵循、RAG 忠实性、数学推理、代码能力、SWE 风格补丁修复、结构化输出、安全边界、Agent 决策、工程可用性等能力。
- 支持规则评分与 LLM-as-a-Judge 裁判评分混合。
- 支持每个模型单独设置超时和并发。
- 自动生成综合排名、能力短板、通过率、延迟、区分度诊断和评测报告。
- 使用 Python 标准库实现，无需复杂依赖。

## 快速启动

```powershell
cd D:\LLM-Study\projects\llm-auto-evaluator
python server.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

## 配置模型接口

复制公开模板为私有配置：

```powershell
Copy-Item config.example.json config.private.json
```

然后编辑 `config.private.json`：

```json
{
  "targets": [
    {
      "name": "example-model",
      "base_url": "https://api.example.com/v1",
      "api_key": "YOUR_API_KEY",
      "model": "example-model",
      "temperature": 0,
      "timeout": 360,
      "concurrency": 1
    }
  ],
  "judge": {
    "name": "judge-model",
    "base_url": "https://api.example.com/v1",
    "api_key": "YOUR_API_KEY",
    "model": "judge-model",
    "temperature": 0,
    "timeout": 360,
    "concurrency": 1
  }
}
```

`config.private.json` 已被 `.gitignore` 忽略，不应提交到 GitHub。

## 测试集

当前内置测试集包括：

- `default_zh.json`：快速通用能力测试集。
- `comprehensive_zh.json`：35 题综合能力测试集，包含高区分题和 SWE-bench-inspired 代码工程题。
- `diagnostic_zh.json`：关键能力诊断测试集。
- `discriminative_zh.json` / `hard_discriminative_zh.json`：区分度增强测试集。
- `selected_discriminative_zh.json`：精选小样本高区分测试集。

其中 `comprehensive_zh.json` 适合对外展示综合评测结果。

## 评分方式

平台使用两类评分：

- 规则评分：基于 JSON 格式、关键词、正则、数值答案、行数、禁用内容等确定性规则。
- 裁判评分：可选启用一个更强模型作为 judge，对开放题、代码题和综合分析题进行 rubric 评分。

启用裁判评分时，调用次数大约为：

```text
参赛模型调用次数 + 成功答案数量对应的裁判调用次数
```

例如 7 个模型、35 道题，最多约 `35 * 7 * 2 = 490` 次调用。

## 报告输出

评测完成后会生成：

- JSON 报告
- Markdown 报告
- HTML 报告

报告默认保存到 `reports/`。该目录已被 `.gitignore` 忽略，因为报告中可能包含接口地址、模型输出和脱敏凭证片段。

## 本地 Mock 测试

如果没有可用模型接口，可以先启动 mock 服务：

```powershell
python scripts\mock_openai_server.py
```

然后在页面中填写：

```text
模型 ID: mock-good
Base URL: http://127.0.0.1:9999/v1
API Key: 留空
```

## 安全注意事项

- 不要提交 `config.private.json`。
- 不要提交 `.env`。
- 不要提交 `reports/`。
- 如果 API Key 曾经出现在聊天记录、截图、日志或报告中，应立即轮换。
- 对外展示报告时建议使用脱敏版。

## 项目结构

```text
llm-auto-evaluator/
├─ datasets/                 # 内置测试集
├─ scripts/                  # 辅助脚本和 mock 服务
├─ static/                   # 前端页面
├─ evaluator.py              # 模型调用、评分、报告生成
├─ server.py                 # 本地 Web 服务
├─ config.example.json       # 公开配置模板
├─ config.private.json       # 私有配置，不提交
└─ reports/                  # 评测报告，不提交
```

## 后续方向

- 增加真实 patch 执行环境，向 SWE-bench 风格自动代码修复评测扩展。
- 增加 token 成本统计。
- 增加历史报告对比。
- 增加自定义测试集导入。
- 增加多轮对话和工具调用评测。
