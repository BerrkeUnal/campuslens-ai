// Değerlendirme ekranı (evaluation.html)
// Sözleşme: startEvaluation, refreshResults, renderComparisonTable

import { METHOD_LABELS, RETRIEVAL_METHODS, getEvaluationResults, runEvaluation } from "./api.js";
import {
  el,
  formatCost,
  formatDate,
  formatDecimal,
  formatMs,
  formatPercent,
  initPage,
  renderError,
  setButtonBusy,
} from "./ui.js";

const SPLIT_LABELS = { validation: "Doğrulama", test: "Test" };
const STATUS_LABELS = { running: "Çalışıyor", completed: "Tamamlandı", failed: "Başarısız" };

// better: "high" → büyük değer iyi, "low" → küçük değer iyi
const METRIC_COLUMNS = [
  { key: "recall_at_5", label: "Source Recall@5", better: "high", format: formatPercent, bar: true },
  { key: "mrr_at_5", label: "MRR@5", better: "high", format: (v) => formatDecimal(v, 2) },
  { key: "answer_correctness", label: "Answer correctness", better: "high", format: (v) => formatDecimal(v, 2) },
  { key: "hallucination_rate", label: "Hallucination", better: "low", format: formatPercent, bar: true },
  { key: "abstention_accuracy", label: "Abstention acc.", better: "high", format: formatPercent },
  { key: "average_latency_ms", label: "Ort. süre", better: "low", format: formatMs },
  { key: "total_cost_usd", label: "Toplam maliyet", better: "low", format: formatCost },
  { key: "total_cases", label: "Soru", better: null, format: (v) => String(v ?? "—") },
];

const POLL_INTERVAL_MS = 3000;
const POLL_MAX_DURATION_MS = 10 * 60 * 1000;

const dom = {};
let running = false;
let refreshing = false;
let pollTimer = null;
let pollStartedAt = 0;
let lastResults = [];

// ---------------------------------------------------------------------------
// Koşu durumu
// ---------------------------------------------------------------------------

/** @param {"idle"|"running"|"completed"|"error"} state */
function setRunState(state, { message, error } = {}) {
  const box = dom.status;
  box.className = "state section-spaced";
  box.hidden = state === "idle";
  setButtonBusy(dom.submit, state === "running", "Çalışıyor…");

  if (state === "running") {
    box.classList.add("state--loading");
    box.replaceChildren(
      el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
      el("span", { text: message }),
    );
  } else if (state === "completed") {
    box.classList.add("state--success");
    box.replaceChildren(el("p", { text: message }));
  } else if (state === "error") {
    box.classList.add("state--error");
    renderError(box, error || { message }, "Değerlendirme başlatılamadı");
  }
}

export async function startEvaluation(event) {
  if (event) event.preventDefault();
  if (running) return;

  const split = dom.split.value;
  const methods = dom.method.value === "all" ? RETRIEVAL_METHODS : [dom.method.value];
  running = true;

  const started = [];
  try {
    for (const [index, method] of methods.entries()) {
      setRunState("running", {
        message: `${METHOD_LABELS[method]} değerlendiriliyor (${index + 1}/${methods.length}, ${SPLIT_LABELS[split]} seti)…`,
      });
      started.push(await runEvaluation({ method, split }));
    }

    const stillRunning = started.some((run) => run.status === "running");
    setRunState(stillRunning ? "running" : "completed", {
      message: stillRunning
        ? "Koşular başlatıldı; sonuçlar hazır oldukça tablo otomatik güncellenecek."
        : `${started.length} koşu tamamlandı.`,
    });
    if (started.some((run) => run.is_mock)) {
      dom.status.append(el("p", { className: "muted", text: "Not: backend şu an mock metrik döndürüyor." }));
    }
  } catch (error) {
    setRunState("error", { error });
    if (started.length) {
      dom.status.append(el("p", { className: "state__detail", text: `${started.length} koşu bu hatadan önce başlatıldı.` }));
    }
  } finally {
    running = false;
    setButtonBusy(dom.submit, false);
    // Görüntülenen seti, yeni koşunun setine eşitle.
    dom.split.value = split;
    await refreshResults();
  }
}

// ---------------------------------------------------------------------------
// Sonuçlar
// ---------------------------------------------------------------------------

function latestCompletedByMethod(results, split) {
  const latest = new Map();
  for (const result of results) {
    if (result.split !== split || result.status !== "completed" || !result.metrics) continue;
    const current = latest.get(result.method);
    if (!current || new Date(result.started_at) > new Date(current.started_at)) {
      latest.set(result.method, result);
    }
  }
  return latest;
}

function bestValues(rows) {
  const best = {};
  for (const column of METRIC_COLUMNS) {
    if (!column.better) continue;
    const values = rows.map((row) => row.metrics[column.key]).filter((v) => typeof v === "number");
    if (values.length < 2) continue;
    best[column.key] = column.better === "high" ? Math.max(...values) : Math.min(...values);
  }
  return best;
}

function metricCell(column, value, best) {
  const cell = el("td", { className: "num", text: column.format(value) });
  if (typeof value === "number" && best[column.key] === value) {
    cell.classList.add("is-best");
    cell.append(el("span", { className: "visually-hidden", text: " (en iyi)" }));
  }
  if (column.bar && typeof value === "number") {
    const width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    const fill = el("span", { className: "bar__fill" });
    fill.style.width = width;
    cell.append(el("span", { className: "bar", attrs: { "aria-hidden": "true" } }, [fill]));
  }
  return cell;
}

/**
 * Seçili soru setinde her yöntemin en son tamamlanan koşusunu yan yana gösterir.
 * @param {Array} results  EvaluationResultResponse listesi
 * @param {string} split
 */
export function renderComparisonTable(results, split = dom.split.value) {
  const latest = latestCompletedByMethod(results, split);
  dom.comparisonSplit.textContent = `${SPLIT_LABELS[split] || split} seti`;

  if (!latest.size) {
    showResultsState("empty");
    return;
  }

  const headRow = el("tr", {}, [
    el("th", { text: "Yöntem", attrs: { scope: "col" } }),
    ...METRIC_COLUMNS.map((column) => el("th", { className: "num", text: column.label, attrs: { scope: "col" } })),
  ]);
  dom.comparisonHead.replaceChildren(headRow);

  const completedRows = [...latest.values()];
  const best = bestValues(completedRows);

  const rows = RETRIEVAL_METHODS.map((method) => {
    const result = latest.get(method);
    const nameCell = el("th", { attrs: { scope: "row" } }, [el("div", { text: METHOD_LABELS[method] })]);

    if (!result) {
      nameCell.append(el("div", { className: "muted", text: "Henüz koşulmadı" }));
      return el("tr", {}, [nameCell, ...METRIC_COLUMNS.map(() => el("td", { className: "num", text: "—" }))]);
    }

    const meta = el("div", { className: "muted", text: formatDate(result.completed_at || result.started_at) });
    if (result.is_mock) meta.append(" ", el("span", { className: "badge badge--mock", text: "mock" }));
    nameCell.append(meta);
    return el("tr", {}, [nameCell, ...METRIC_COLUMNS.map((column) => metricCell(column, result.metrics[column.key], best))]);
  });

  dom.comparisonBody.replaceChildren(...rows);
  showResultsState("table");
}

function renderHistory(results) {
  const sorted = results.slice().sort((a, b) => new Date(b.started_at) - new Date(a.started_at));
  dom.historyEmpty.hidden = sorted.length > 0;
  dom.historyWrap.hidden = sorted.length === 0;

  dom.historyBody.replaceChildren(
    ...sorted.map((result) =>
      el("tr", {}, [
        el("td", { text: formatDate(result.started_at) }),
        el("td", { text: METHOD_LABELS[result.method] || result.method }),
        el("td", { text: SPLIT_LABELS[result.split] || result.split }),
        el("td", {}, [
          el("span", { className: `badge badge--${result.status}`, text: STATUS_LABELS[result.status] || result.status }),
        ]),
        el("td", { className: "num", text: result.metrics?.total_cases ?? "—" }),
        el("td", { className: "num", text: formatPercent(result.metrics?.recall_at_5) }),
      ]),
    ),
  );
}

function showResultsState(state) {
  dom.resultsLoading.hidden = state !== "loading";
  dom.resultsError.hidden = state !== "error";
  dom.resultsEmpty.hidden = state !== "empty";
  dom.comparisonWrap.hidden = state !== "table";
}

function schedulePolling(results) {
  clearTimeout(pollTimer);
  pollTimer = null;
  const hasRunning = results.some((result) => result.status === "running");
  if (!hasRunning) {
    pollStartedAt = 0;
    return;
  }
  if (!pollStartedAt) pollStartedAt = Date.now();
  if (Date.now() - pollStartedAt > POLL_MAX_DURATION_MS) {
    pollStartedAt = 0;
    return;
  }
  pollTimer = setTimeout(() => refreshResults({ silent: true }), POLL_INTERVAL_MS);
}

export async function refreshResults({ silent = false } = {}) {
  if (refreshing) return;
  refreshing = true;
  if (!silent) {
    setButtonBusy(dom.refresh, true, "Yenileniyor…");
    if (dom.comparisonWrap.hidden) showResultsState("loading");
  }

  try {
    const { results = [] } = (await getEvaluationResults()) || {};
    lastResults = results;
    renderComparisonTable(results);
    renderHistory(results);
    schedulePolling(results);
  } catch (error) {
    clearTimeout(pollTimer);
    renderError(dom.resultsError, error, "Sonuçlar alınamadı");
    showResultsState("error");
  } finally {
    refreshing = false;
    setButtonBusy(dom.refresh, false);
  }
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

function init() {
  initPage();

  const byId = (id) => document.getElementById(id);
  Object.assign(dom, {
    form: byId("evaluation-form"),
    method: byId("eval-method"),
    split: byId("eval-split"),
    submit: byId("evaluation-submit"),
    refresh: byId("results-refresh"),
    status: byId("evaluation-status"),
    comparisonSplit: byId("comparison-split"),
    resultsLoading: byId("results-loading"),
    resultsError: byId("results-error"),
    resultsEmpty: byId("results-empty"),
    comparisonWrap: byId("comparison-wrap"),
    comparisonHead: byId("comparison-head"),
    comparisonBody: byId("comparison-body"),
    historyEmpty: byId("history-empty"),
    historyWrap: byId("history-wrap"),
    historyBody: byId("history-body"),
  });

  dom.form.addEventListener("submit", startEvaluation);
  dom.refresh.addEventListener("click", () => refreshResults());
  // Soru seti değişince tabloyu yeniden çiz (yeni istek gerekmez).
  dom.split.addEventListener("change", () => renderComparisonTable(lastResults));

  refreshResults();
}

init();
