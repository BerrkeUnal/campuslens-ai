// Öğrenci soru-cevap ekranı (index.html)
// Sözleşme: submitQuestion, renderAnswer, renderSources, setChatState
// Öğrenci yöntem seçmez; tüm sorular hybrid_rerank ile gönderilir.

import { askQuestion } from "./api.js";
import { el, externalLink, formatMs, formatScore, initPage, renderError, setButtonBusy } from "./ui.js";

const STUDENT_METHOD = "hybrid_rerank";
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

function sourceElementId(sourceId) {
  return `source-${sourceId}`;
}

function focusSource(sourceId) {
  const card = document.getElementById(sourceElementId(sourceId));
  if (!card) return;
  document.querySelectorAll(".source-card.is-highlighted").forEach((node) => node.classList.remove("is-highlighted"));
  card.classList.add("is-highlighted");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.focus({ preventScroll: true });
}

/** Yanıt metnini [S1] işaretlerini tıklanabilir yaparak güvenli biçimde üretir. */
function buildAnswerNodes(answer, sources) {
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
          "aria-controls": sourceElementId(sourceId),
        },
      });
      button.addEventListener("click", () => focusSource(sourceId));
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
  return [el("div", {}, [el("dt", { text: "Yanıt süresi" }), el("dd", { text: formatMs(metrics.total_latency_ms) })])];
}

export function renderAnswer(response) {
  const sources = Array.isArray(response.sources) ? response.sources : [];
  const insufficient = Boolean(response.insufficient_context);

  setChatState(insufficient ? "insufficient" : "success");

  dom.answerTitle.textContent = insufficient ? "Sistemin yanıtı" : "Yanıt";
  dom.answerText.replaceChildren(...buildAnswerNodes(response.answer || "", sources));
  dom.answerMetrics.replaceChildren(...buildMetrics(response.metrics));

  renderSources(sources, dom.sourceList, {
    countNode: dom.sourcesCount,
    emptyNode: dom.sourcesEmpty,
    insufficient,
  });
}

function buildSourceCard(source) {
  const scores = [el("li", { className: "score", text: `retrieval ${formatScore(source.retrieval_score)}` })];
  if (typeof source.rerank_score === "number") {
    scores.push(el("li", { className: "score", text: `rerank ${formatScore(source.rerank_score)}` }));
  }

  return el(
    "li",
    {
      className: "source-card",
      attrs: { id: sourceElementId(source.source_id), tabindex: "-1" },
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
export function renderSources(sources, container, { countNode, emptyNode, insufficient = false } = {}) {
  container.replaceChildren(...sources.map(buildSourceCard));

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

export async function submitQuestion(event) {
  if (event) event.preventDefault();
  if (inFlight) return;

  const question = readQuestion();
  if (!question) return;

  inFlight = true;
  setButtonBusy(dom.submit, true, "Yanıtlanıyor…");
  setChatState("loading");

  try {
    const response = await askQuestion(question, STUDENT_METHOD);
    renderAnswer(response);
  } catch (error) {
    setChatState("error", { error });
  } finally {
    inFlight = false;
    setButtonBusy(dom.submit, false);
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
    submit: byId("ask-submit"),
    empty: byId("chat-empty"),
    loading: byId("chat-loading"),
    error: byId("chat-error"),
    answer: byId("chat-answer"),
    insufficient: byId("chat-insufficient"),
    answerTitle: byId("answer-title"),
    answerText: byId("answer-text"),
    answerMetrics: byId("answer-metrics"),
    sourceList: byId("source-list"),
    sourcesCount: byId("sources-count"),
    sourcesEmpty: byId("sources-empty"),
  });

  dom.form.addEventListener("submit", submitQuestion);
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
