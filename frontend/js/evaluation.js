// Değerlendirme ekranı (evaluation.html)
// Sözleşme: startEvaluation, refreshResults, renderComparisonTable

import { initAdminShell } from "./admin-shell.js";
import { METHOD_LABELS, RETRIEVAL_METHODS, getEvaluationResults, runEvaluation } from "./api.js";
import {
  el,
  formatCost,
  formatDate,
  formatDecimal,
  formatMs,
  formatPercent,
  icon,
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

/** @param {"idle"|"running"|"completed"|"info"|"error"} state */
function setRunState(state, { message, error } = {}) {
  const box = dom.status;
  box.className = "section-spaced";
  box.removeAttribute("role");
  box.hidden = state === "idle";
  setButtonBusy(dom.submit, state === "running", "Çalışıyor…");

  if (state === "running") {
    box.classList.add("notice", "notice--info");
    box.replaceChildren(
      el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
      el("span", { text: message }),
    );
  } else if (state === "completed") {
    box.classList.add("notice", "notice--success");
    box.replaceChildren(icon("check-circle"), el("div", {}, [el("p", { text: message })]));
  } else if (state === "info") {
    box.classList.add("notice", "notice--warning");
    box.replaceChildren(
      icon("info"),
      el("div", {}, [el("p", { className: "notice__title", text: message.title }), el("p", { text: message.text })]),
    );
  } else if (state === "error") {
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

    if (started.some((run) => run.is_mock)) {
      // Backend örnek (sabit) metrik döndürdü: bunlar gerçek ölçüm değildir, gösterilmez.
      setRunState("info", {
        message: {
          title: "Gerçek değerlendirme sonucu yok",
          text:
            "Değerlendirme servisi henüz gerçek sonuç üretmiyor (backend örnek veri döndürdü). " +
            "Bu koşuya ait sayılar gösterilmeyecek; gerçek değerlendirme etkinleştirildiğinde sonuçlar burada görünecek.",
        },
      });
      return;
    }
    const stillRunning = started.some((run) => run.status === "running");
    const failed = started.filter((run) => run.status === "failed").length;
    setRunState(stillRunning ? "running" : "completed", {
      message: stillRunning
        ? "Koşular başlatıldı; sonuçlar hazır oldukça tablo otomatik güncellenecek."
        : failed
          ? `${started.length} koşudan ${failed} tanesi başarısız oldu. Ayrıntı için koşu geçmişine bakın.`
          : `${started.length} koşu tamamlandı.`,
    });
  } catch (error) {
    setRunState("error", { error });
    if (started.length) {
      dom.status.lastElementChild.append(el("p", { className: "notice__detail", text: `${started.length} koşu bu hatadan önce başlatıldı.` }));
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
    if (result.is_mock || result.split !== split || result.status !== "completed" || !result.metrics) continue;
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
  if (typeof value !== "number") cell.title = "Bu metrik bu koşuda hesaplanmadı.";
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
    renderEmptyState(results, split);
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

    nameCell.append(el("div", { className: "muted", text: formatDate(result.completed_at || result.started_at) }));
    return el("tr", {}, [nameCell, ...METRIC_COLUMNS.map((column) => metricCell(column, result.metrics[column.key], best))]);
  });

  dom.comparisonBody.replaceChildren(...rows);
  showResultsState("table");
}

/** Gerçek sonuç yoksa sayı yerine açıklayıcı boş durum gösterilir. */
function renderEmptyState(results, split) {
  const realRuns = realResults(results);
  const anyCompleted = realRuns.some((result) => result.status === "completed" && result.metrics);
  const anyRunning = realRuns.some((result) => result.status === "running");

  if (anyCompleted) {
    dom.emptyTitle.textContent = `${SPLIT_LABELS[split] || split} seti için tamamlanmış değerlendirme yok`;
    dom.emptyText.textContent = "Bu soru seti için bir koşu başlatın veya diğer seti seçin.";
  } else if (anyRunning) {
    dom.emptyTitle.textContent = "Değerlendirme çalışıyor";
    dom.emptyText.textContent = "Koşu tamamlandığında metrikler burada görünecek.";
  } else {
    dom.emptyTitle.textContent = "Henüz değerlendirme çalıştırılmadı";
    dom.emptyText.textContent =
      "Metrikler yalnızca gerçek bir değerlendirme koşusu tamamlandığında gösterilir. Yukarıdan bir koşu başlatın.";
  }

  const mockCount = results.length - realRuns.length;
  dom.emptyNote.hidden = mockCount === 0;
  dom.emptyNote.textContent = mockCount
    ? `Backend'in döndürdüğü ${mockCount} örnek (mock) koşu gerçek ölçüm olmadığı için gösterilmiyor.`
    : "";
}

/** Backend'in is_mock olarak işaretlediği sonuçlar ekranda hiç kullanılmaz. */
function realResults(results) {
  return results.filter((result) => !result.is_mock);
}

function statusCell(result) {
  const cell = el("td", {}, [
    el("span", { className: `pill pill--${result.status}` }, [
      el("span", { className: "pill__dot", attrs: { "aria-hidden": "true" } }),
      STATUS_LABELS[result.status] || result.status,
    ]),
  ]);
  if (result.status === "failed") {
    cell.append(
      el("p", {
        className: "history__error",
        text: result.error_message || "Koşu başarısız oldu; backend bir açıklama döndürmedi.",
      }),
    );
  }
  return cell;
}

function renderHistory(results) {
  const sorted = realResults(results).sort((a, b) => new Date(b.started_at) - new Date(a.started_at));
  dom.historyEmpty.hidden = sorted.length > 0;
  dom.historyWrap.hidden = sorted.length === 0;

  dom.historyBody.replaceChildren(
    ...sorted.map((result) =>
      el("tr", {}, [
        el("td", { text: formatDate(result.started_at) }),
        el("td", { text: METHOD_LABELS[result.method] || result.method }),
        el("td", { text: SPLIT_LABELS[result.split] || result.split }),
        statusCell(result),
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
  const hasRunning = realResults(results).some((result) => result.status === "running");
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
  if (!initAdminShell()) return;

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
    emptyTitle: byId("results-empty-title"),
    emptyText: byId("results-empty-text"),
    emptyNote: byId("results-empty-note"),
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
