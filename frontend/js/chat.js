// Soru-cevap ekranı (index.html)
// Sözleşme: submitQuestion, renderAnswer, renderSources, setChatState

import { METHOD_LABELS, RETRIEVAL_METHODS, askQuestion } from "./api.js";
import {
  el,
  externalLink,
  formatCost,
  formatMs,
  formatScore,
  initPage,
  renderError,
  setButtonBusy,
} from "./ui.js";

const QUESTION_MIN = 3;
const QUESTION_MAX = 1000;
const CITATION_PATTERN = /\[(S\d+)\]/g;

const dom = {};
let inFlight = false;

// ---------------------------------------------------------------------------
// Durum yönetimi
// ---------------------------------------------------------------------------

/**
 * @param {"empty"|"loading"|"success"|"insufficient"|"error"} state
 * @param {{error?: Error}} [options]
 */
export function setChatState(state, { error } = {}) {
  dom.empty.hidden = state !== "empty";
  dom.loading.hidden = state !== "loading";
  dom.error.hidden = state !== "error";
  dom.answer.hidden = state !== "success" && state !== "insufficient";
  dom.insufficient.hidden = state !== "insufficient";

  if (state === "error") {
    renderError(dom.error, error, "Yanıt alınamadı");
  }

  if (state !== "success" && state !== "insufficient") {
    dom.sourceList.replaceChildren();
    dom.sourcesCount.textContent = "";
    dom.sourcesEmpty.hidden = false;
    dom.sourcesEmpty.textContent =
      state === "loading" ? "Kaynaklar aranıyor…" : "Yanıtta kullanılan belge parçaları burada listelenir.";
  }
}

// ---------------------------------------------------------------------------
// Yanıt ve kaynak gösterimi
// ---------------------------------------------------------------------------

function sourceElementId(scope, sourceId) {
  return `source-${scope}-${sourceId}`;
}

function focusSource(scope, sourceId) {
  const card = document.getElementById(sourceElementId(scope, sourceId));
  if (!card) return;
  document.querySelectorAll(".source-card.is-highlighted").forEach((node) => node.classList.remove("is-highlighted"));
  card.classList.add("is-highlighted");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.focus({ preventScroll: true });
}

/** Yanıt metnini [S1] işaretlerini tıklanabilir yaparak güvenli biçimde üretir. */
function buildAnswerNodes(answer, sources, scope) {
  const knownIds = new Set(sources.map((source) => source.source_id));
  const nodes = [];
  let lastIndex = 0;

  for (const match of answer.matchAll(CITATION_PATTERN)) {
    if (match.index > lastIndex) nodes.push(document.createTextNode(answer.slice(lastIndex, match.index)));
    const sourceId = match[1];

    if (knownIds.has(sourceId)) {
      const button = el("button", {
        className: "citation",
        text: sourceId,
        attrs: {
          type: "button",
          "aria-label": `${sourceId} kaynağına git`,
          "aria-controls": sourceElementId(scope, sourceId),
        },
      });
      button.addEventListener("click", () => focusSource(scope, sourceId));
      nodes.push(button);
    } else {
      nodes.push(
        el("span", {
          className: "citation citation--missing",
          text: sourceId,
          attrs: { title: "Bu kaynak yanıtla birlikte gelmedi" },
        }),
      );
    }
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < answer.length) nodes.push(document.createTextNode(answer.slice(lastIndex)));
  return nodes;
}

function buildMetrics(metrics = {}) {
  const items = [
    ["Toplam süre", formatMs(metrics.total_latency_ms)],
    ["Arama", formatMs(metrics.retrieval_latency_ms)],
    ["Üretim", formatMs(metrics.generation_latency_ms)],
    ["Tahmini maliyet", formatCost(metrics.estimated_cost_usd)],
  ];
  return items.map(([label, value]) => el("div", {}, [el("dt", { text: label }), el("dd", { text: value })]));
}

export function renderAnswer(response) {
  const sources = Array.isArray(response.sources) ? response.sources : [];
  const insufficient = Boolean(response.insufficient_context);

  setChatState(insufficient ? "insufficient" : "success");

  dom.answerTitle.textContent = insufficient ? "Sistemin yanıtı" : "Yanıt";
  dom.answerMethod.textContent = METHOD_LABELS[response.method] || response.method || "";
  dom.answerText.replaceChildren(...buildAnswerNodes(response.answer || "", sources, "main"));
  dom.answerMetrics.replaceChildren(...buildMetrics(response.metrics));

  renderSources(sources, dom.sourceList, {
    scope: "main",
    countNode: dom.sourcesCount,
    emptyNode: dom.sourcesEmpty,
    insufficient,
  });
}

function buildSourceCard(source, scope) {
  const scores = [el("li", { className: "score", text: `retrieval ${formatScore(source.retrieval_score)}` })];
  if (typeof source.rerank_score === "number") {
    scores.push(el("li", { className: "score", text: `rerank ${formatScore(source.rerank_score)}` }));
  }

  return el(
    "li",
    {
      className: "source-card",
      attrs: { id: sourceElementId(scope, source.source_id), tabindex: "-1" },
    },
    [
      el("div", { className: "source-card__head" }, [
        el("span", { className: "source-card__id", text: source.source_id }),
        el("p", { className: "source-card__title" }, [externalLink(source.source_url, source.document_title)]),
      ]),
      el("div", { className: "source-card__page", text: `Sayfa ${source.page_number}` }),
      el("blockquote", { className: "source-card__excerpt", text: source.excerpt }),
      el("ul", { className: "scores", attrs: { "aria-label": "Skorlar" } }, scores),
    ],
  );
}

/**
 * @param {Array} sources  AskResponse.sources
 * @param {HTMLElement} container  <ol> veya <ul>
 */
export function renderSources(sources, container, { scope = "main", countNode, emptyNode, insufficient = false } = {}) {
  container.replaceChildren(...sources.map((source) => buildSourceCard(source, scope)));

  if (countNode) countNode.textContent = sources.length ? `${sources.length} parça` : "";
  if (emptyNode) {
    emptyNode.hidden = sources.length > 0 && !insufficient;
    if (!sources.length) {
      emptyNode.textContent = "Bu yanıt için kaynak parça döndürülmedi.";
    } else if (insufficient) {
      emptyNode.textContent = "Aşağıdaki parçalar incelendi ancak soruyu yanıtlamak için yeterli bulunmadı.";
    }
  }
}

// ---------------------------------------------------------------------------
// Form gönderimi
// ---------------------------------------------------------------------------

function readQuestion() {
  const question = dom.question.value.trim();
  let message = "";
  if (question.length < QUESTION_MIN) message = `Soru en az ${QUESTION_MIN} karakter olmalıdır.`;
  else if (question.length > QUESTION_MAX) message = `Soru en fazla ${QUESTION_MAX} karakter olabilir.`;

  dom.questionError.hidden = !message;
  dom.questionError.textContent = message;
  dom.question.setAttribute("aria-invalid", String(Boolean(message)));
  if (message) {
    dom.question.focus();
    return null;
  }
  return question;
}

function setFormBusy(busy, activeButton) {
  inFlight = busy;
  setButtonBusy(dom.submit, busy && activeButton === dom.submit, "Yanıtlanıyor…");
  setButtonBusy(dom.compare, busy && activeButton === dom.compare, "Karşılaştırılıyor…");
  if (busy) {
    dom.submit.disabled = true;
    dom.compare.disabled = true;
  }
}

export async function submitQuestion(event) {
  if (event) event.preventDefault();
  if (inFlight) return;

  const question = readQuestion();
  if (!question) return;

  const method = dom.method.value;
  dom.compareSection.hidden = true;
  setFormBusy(true, dom.submit);
  setChatState("loading");

  try {
    const response = await askQuestion(question, method);
    renderAnswer(response);
  } catch (error) {
    setChatState("error", { error });
  } finally {
    setFormBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Aynı soruyu dört yöntemle yan yana karşılaştırma
// ---------------------------------------------------------------------------

function buildCompareColumn(method, outcome) {
  const header = el("h3", {}, [el("span", { text: METHOD_LABELS[method] })]);
  const column = el("article", { className: "compare-column", attrs: { "aria-label": METHOD_LABELS[method] } }, [header]);

  if (outcome.status === "rejected") {
    const box = el("div", { className: "state state--error" });
    renderError(box, outcome.reason, "Yanıt alınamadı");
    column.append(box);
    return column;
  }

  const response = outcome.value;
  const sources = Array.isArray(response.sources) ? response.sources : [];
  const scope = `cmp-${method}`;
  header.append(el("span", { className: "muted", text: formatMs(response.metrics?.total_latency_ms) }));

  if (response.insufficient_context) {
    column.append(el("div", { className: "state state--neutral" }, [el("p", { text: "Yeterli bilgi bulunamadı." })]));
  }

  column.append(
    el("p", { className: "answer__text" }, buildAnswerNodes(response.answer || "", sources, scope)),
    el("p", { className: "muted", text: sources.length ? `${sources.length} kaynak parça` : "Kaynak yok" }),
  );
  const list = el("ol", { className: "source-list" });
  renderSources(sources, list, { scope });
  column.append(list);
  return column;
}

async function compareMethods() {
  if (inFlight) return;
  const question = readQuestion();
  if (!question) return;

  setFormBusy(true, dom.compare);
  setChatState("empty");
  dom.empty.hidden = true;
  dom.compareSection.hidden = false;
  dom.compareQuestion.textContent = `“${question}”`;
  dom.compareGrid.replaceChildren(
    el("div", { className: "state state--loading" }, [
      el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
      el("span", { text: "Dört yöntem aynı anda deneniyor…" }),
    ]),
  );

  try {
    const outcomes = await Promise.allSettled(RETRIEVAL_METHODS.map((method) => askQuestion(question, method)));
    dom.compareGrid.replaceChildren(...RETRIEVAL_METHODS.map((method, i) => buildCompareColumn(method, outcomes[i])));
    dom.compareSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } finally {
    setFormBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

function updateCounter() {
  dom.counter.textContent = `${dom.question.value.length} / ${QUESTION_MAX}`;
  if (dom.question.getAttribute("aria-invalid") === "true" && dom.question.value.trim().length >= QUESTION_MIN) {
    dom.question.setAttribute("aria-invalid", "false");
    dom.questionError.hidden = true;
  }
}

function init() {
  initPage();

  const byId = (id) => document.getElementById(id);
  Object.assign(dom, {
    form: byId("ask-form"),
    question: byId("question"),
    questionError: byId("question-error"),
    counter: byId("question-counter"),
    method: byId("method"),
    submit: byId("ask-submit"),
    compare: byId("compare-submit"),
    empty: byId("chat-empty"),
    loading: byId("chat-loading"),
    error: byId("chat-error"),
    answer: byId("chat-answer"),
    insufficient: byId("chat-insufficient"),
    answerTitle: byId("answer-title"),
    answerMethod: byId("answer-method"),
    answerText: byId("answer-text"),
    answerMetrics: byId("answer-metrics"),
    sourceList: byId("source-list"),
    sourcesCount: byId("sources-count"),
    sourcesEmpty: byId("sources-empty"),
    compareSection: byId("compare-section"),
    compareGrid: byId("compare-grid"),
    compareQuestion: byId("compare-question"),
  });

  dom.form.addEventListener("submit", submitQuestion);
  dom.compare.addEventListener("click", compareMethods);
  dom.question.addEventListener("input", updateCounter);
  dom.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      dom.form.requestSubmit();
    }
  });

  document.querySelectorAll("[data-example]").forEach((chip) => {
    chip.addEventListener("click", () => {
      dom.question.value = chip.textContent.trim();
      updateCounter();
      dom.question.focus();
    });
  });

  setChatState("empty");
}

init();
