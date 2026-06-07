const modelList = document.querySelector("#modelList");
const modelTemplate = document.querySelector("#modelTemplate");
const addModelBtn = document.querySelector("#addModelBtn");
const runBtn = document.querySelector("#runBtn");
const loadPrivateBtn = document.querySelector("#loadPrivateBtn");
const loadDemoBtn = document.querySelector("#loadDemoBtn");
const datasetSelect = document.querySelector("#datasetSelect");
const caseLimit = document.querySelector("#caseLimit");
const temperature = document.querySelector("#temperature");
const judgeEnabled = document.querySelector("#judgeEnabled");
const judgeInfo = document.querySelector("#judgeInfo");
const globalStatus = document.querySelector("#globalStatus");
const progressText = document.querySelector("#progressText");
const progressFill = document.querySelector("#progressFill");
const spinner = document.querySelector("#spinner");
const result = document.querySelector("#result");
const rankingBody = document.querySelector("#rankingBody");
const separationBox = document.querySelector("#separationBox");
const modelCards = document.querySelector("#modelCards");
const insights = document.querySelector("#insights");
const reportMeta = document.querySelector("#reportMeta");
const jsonReportLink = document.querySelector("#jsonReportLink");
const htmlReportLink = document.querySelector("#htmlReportLink");
const mdReportLink = document.querySelector("#mdReportLink");

let pollTimer = null;
let datasets = [];
let judgeConfig = null;

function addModel(values = {}) {
  if (modelList.children.length >= 10) {
    setStatus("最多支持 10 个接口", "warn");
    return;
  }
  const node = modelTemplate.content.cloneNode(true);
  const card = node.querySelector(".model-card");
  card.querySelectorAll("input").forEach((input) => {
    input.value = values[input.dataset.field] || "";
  });
  const keyInput = card.querySelector('[data-field="api_key"]');
  if (values.has_server_key) {
    keyInput.dataset.serverKey = "1";
    keyInput.value = values.api_key || "服务器端已保存";
    keyInput.readOnly = true;
  }
  card.querySelector(".remove-model").addEventListener("click", () => {
    if (modelList.children.length > 1) {
      card.remove();
    }
  });
  modelList.appendChild(node);
}

function getTargets() {
  return Array.from(modelList.querySelectorAll(".model-card"))
    .map((card) => {
      const target = {};
      card.querySelectorAll("input").forEach((input) => {
        if (input.dataset.field === "api_key" && input.dataset.serverKey === "1") {
          return;
        }
        target[input.dataset.field] = input.value.trim();
      });
      target.temperature = Number(temperature.value || 0);
      return target;
    })
    .filter((target) => target.base_url || target.model || target.api_key || target.name);
}

function setStatus(text, type = "normal") {
  globalStatus.textContent = text;
  globalStatus.style.color =
    type === "error" ? "#b42318" : type === "ok" ? "#16794c" : type === "warn" ? "#b35c00" : "#697586";
}

async function loadDataset() {
  const response = await fetch("/api/datasets");
  const data = await response.json();
  datasets = data.datasets || [];
  datasetSelect.innerHTML = datasets
    .map((dataset) => `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)} · ${dataset.case_count} 题</option>`)
    .join("");
  datasetSelect.value = "default_zh";
  renderDatasetSummary();
}

function renderDatasetSummary() {
  const dataset = datasets.find((item) => item.id === datasetSelect.value) || datasets[0];
  if (!dataset) return;
  document.querySelector("#datasetDesc").textContent = dataset.description;
  document.querySelector("#caseCount").textContent = dataset.case_count;
  const categories = dataset.categories || [];
  document.querySelector("#categoryCount").textContent = categories.length;
  document.querySelector("#categoryList").innerHTML = categories
    .map((category) => `<span class="tag">${escapeHtml(category)}</span>`)
    .join("");
  const cases = dataset.cases || [];
  document.querySelector("#casePreviewList").innerHTML = cases
    .map(
      (item, index) => `
        <div class="preview-case">
          <strong>
            <span>${index + 1}. ${escapeHtml(item.id)} · ${escapeHtml(item.category)} · ${escapeHtml(item.difficulty)}</span>
            <span>权重 ${escapeHtml(item.weight ?? 1)}</span>
          </strong>
          <p>${escapeHtml(item.prompt)}</p>
        </div>
      `,
    )
    .join("");
}

async function loadPrivateTargets() {
  try {
    const response = await fetch("/api/config-targets");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "读取私有配置失败");
    if (!data.targets.length) {
      setStatus("未找到私有配置", "warn");
      progressText.textContent = "没有找到 config.private.json 或 targets 为空。";
      return;
    }
    modelList.innerHTML = "";
    data.targets.forEach((target) => addModel(target));
    setStatus(`已加载 ${data.targets.length} 个配置`, "ok");
    progressText.textContent = "已加载服务器端私有配置，API Key 不会暴露在浏览器表单中。";
  } catch (error) {
    setStatus("加载失败", "error");
    progressText.textContent = error.message;
  }
}

async function loadJudgeConfig() {
  try {
    const response = await fetch("/api/judge-config");
    const data = await response.json();
    judgeConfig = data.judge || null;
    if (judgeConfig) {
      judgeInfo.textContent = `${judgeConfig.name} · ${judgeConfig.model}`;
      judgeEnabled.disabled = false;
    } else {
      judgeInfo.textContent = "裁判模型未配置";
      judgeEnabled.checked = false;
      judgeEnabled.disabled = true;
    }
  } catch (error) {
    judgeInfo.textContent = "裁判配置读取失败";
    judgeEnabled.checked = false;
    judgeEnabled.disabled = true;
  }
}

async function startEvaluation() {
  const targets = getTargets();
  if (!targets.length) {
    setStatus("请先输入接口", "error");
    progressText.textContent = "至少需要一个模型接口。";
    return;
  }
  if (targets.length > 10) {
    setStatus("接口数量超限", "error");
    progressText.textContent = "最多支持同时输入 10 个模型。";
    return;
  }
  const missing = targets.find((target) => !target.base_url || !target.model);
  if (missing) {
    setStatus("配置不完整", "error");
    progressText.textContent = "每个模型至少需要 Base URL 和模型 ID。";
    return;
  }

  result.classList.add("hidden");
  spinner.classList.remove("hidden");
  progressFill.style.width = "35%";
  progressText.textContent = "正在提交评测任务...";
  setStatus("评测中", "warn");
  runBtn.disabled = true;

  try {
    const response = await fetch("/api/evaluations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        targets,
        dataset_id: datasetSelect.value,
        case_limit: caseLimit.value ? Number(caseLimit.value) : null,
        judge_enabled: Boolean(judgeEnabled?.checked),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "提交失败");
    pollJob(data.job_id);
  } catch (error) {
    runBtn.disabled = false;
    spinner.classList.add("hidden");
    progressFill.style.width = "0%";
    progressText.textContent = error.message;
    setStatus("提交失败", "error");
  }
}

function pollJob(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    progressText.textContent = job.progress || "正在执行...";
    if (job.status === "running") {
      const current = Number.parseFloat(progressFill.style.width || "35");
      progressFill.style.width = `${Math.min(current + 6, 88)}%`;
      return;
    }
    clearInterval(pollTimer);
    runBtn.disabled = false;
    spinner.classList.add("hidden");

    if (job.status === "completed") {
      progressFill.style.width = "100%";
      setStatus("评测完成", "ok");
      renderReport(job.result);
      return;
    }
    progressFill.style.width = "0%";
    setStatus("评测失败", "error");
    progressText.textContent = job.error || "评测失败，请检查 API 地址、Key 或模型 ID。";
  }, 1200);
}

function renderReport(report) {
  result.classList.remove("hidden");
  reportMeta.textContent = `${report.dataset.name} · ${report.dataset.case_count} 题 · ${report.created_at}`;
  jsonReportLink.href = `/api/reports/${report.id}/json`;
  htmlReportLink.href = `/api/reports/${report.id}/html`;
  mdReportLink.href = `/api/reports/${report.id}/markdown`;

  rankingBody.innerHTML = report.rankings
    .map(
      (row) => `
        <tr>
          <td>${row.rank}</td>
          <td><strong>${escapeHtml(row.name)}</strong><br><span class="muted">${escapeHtml(row.model)}</span></td>
          <td><span class="score-badge">${row.overall_score}</span></td>
          <td>${escapeHtml(row.grade)}</td>
          <td>${row.pass_rate}%</td>
          <td>${row.avg_latency_ms ?? "-"} ms</td>
        </tr>
      `,
    )
    .join("");

  if (report.separation) {
    separationBox.textContent = `${report.separation.level}：最高最低分差 ${report.separation.score_range}，第一第二分差 ${report.separation.top_gap}，标准差 ${report.separation.stddev}。`;
  } else {
    separationBox.textContent = "";
  }

  modelCards.innerHTML = report.models.map(renderModelCard).join("");
  insights.innerHTML = report.insights.map((item) => `<div class="insight">${escapeHtml(item)}</div>`).join("");
  document.querySelector("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderModelCard(model) {
  const summary = model.summary;
  const categoryBars = Object.entries(summary.category_scores)
    .map(
      ([name, score]) => `
        <div class="bar-row">
          <span>${escapeHtml(name)}</span>
          <div class="bar"><span style="width:${Math.max(3, score)}%"></span></div>
          <strong>${score}</strong>
        </div>
      `,
    )
    .join("");
  const weakCases = model.cases
    .filter((item) => item.score < 85 || item.error)
    .slice(0, 4)
    .map(
      (item) => `
        <div class="case-item">
          <strong><span>${escapeHtml(item.case_id)} · ${escapeHtml(item.category)}</span><span>${item.score}</span></strong>
          <p>${escapeHtml(item.error || item.failed.slice(0, 2).join("；") || "无明显扣分点")}</p>
        </div>
      `,
    )
    .join("");

  return `
    <article class="model-result">
      <div class="model-result-head">
        <div>
          <h4>${escapeHtml(model.name)}</h4>
          <p>${escapeHtml(model.model)} · ${summary.success_count}/${summary.case_count} 调用成功</p>
        </div>
        <span class="score-badge">${summary.overall_score}</span>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>分类</span><strong>${escapeHtml(summary.grade.split("：")[0])}</strong></div>
        <div class="metric"><span>通过率</span><strong>${summary.pass_rate}%</strong></div>
        <div class="metric"><span>延迟</span><strong>${summary.avg_latency_ms ?? "-"}ms</strong></div>
      </div>
      <div class="category-score">${categoryBars}</div>
      <div class="case-list">
        ${weakCases || '<div class="case-item"><strong>未发现待加强题目</strong><p>该模型在当前测试集上表现稳定。</p></div>'}
      </div>
    </article>
  `;
}

function fillDemo() {
  modelList.innerHTML = "";
  addModel({
    name: "本地 Ollama 示例",
    base_url: "http://localhost:11434/v1",
    model: "qwen2.5:7b",
    api_key: "",
  });
  addModel({
    name: "OpenAI 兼容接口示例",
    base_url: "https://api.example.com/v1",
    model: "your-model-name",
    api_key: "替换为你的 API Key",
  });
  setStatus("已填入示例", "normal");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

addModelBtn.addEventListener("click", () => addModel());
runBtn.addEventListener("click", startEvaluation);
loadDemoBtn.addEventListener("click", fillDemo);
loadPrivateBtn.addEventListener("click", loadPrivateTargets);
datasetSelect.addEventListener("change", renderDatasetSummary);

addModel();
loadDataset().catch((error) => {
  document.querySelector("#datasetDesc").textContent = error.message;
});
loadJudgeConfig();
