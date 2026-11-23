// Öğrenci sohbet ekranı (index.html)
// Sözleşme: submitQuestion, renderAnswer, renderSources, setChatState
// Öğrenci yöntem seçmez; tüm sorular hybrid_rerank ile gönderilir.
// Backend her soruyu bağımsız yanıtlar; sohbet akışı yalnızca bu sekmede, bellekte tutulur.

import { askQuestion } from "./api.js";
import { el, externalLink, formatMs, icon, initStatus, renderError } from "./ui.js";

const STUDENT_METHOD = "hybrid_rerank";
const QUESTION_MIN = 3;
const QUESTION_MAX = 1000;
const COUNTER_FROM = 800;
const COMPOSER_MAX_HEIGHT = 180;
const CITATION_PATTERN = /\[(S\d+)\]/g;

const dom = {};
const answers = new Map(); // mesaj kimliği -> AskResponse
let inFlight = false;
let conversation = 0; // "Yeni sohbet" ile artar; eski sohbetten gelen geç yanıtlar yok sayılır
let messageSeq = 0;
let activeMessageId = null;

// ---------------------------------------------------------------------------
// Durum
// ---------------------------------------------------------------------------

/** @param {"empty"|"busy"|"idle"} state */
export function setChatState(state) {
  inFlight = state === "busy";
  dom.submit.disabled = inFlight;
  dom.submit.setAttribute("aria-busy", String(inFlight));
  document.querySelectorAll("[data-question]").forEach((button) => (button.disabled = inFlight));

  if (state === "empty") {
    answers.clear();
    activeMessageId = null;
    dom.thread.replaceChildren(dom.welcome);
    dom.welcome.hidden = false;
    renderSources([], {});
  }
}

function scrollThreadToEnd() {
  const scroller = window.matchMedia("(max-width: 960px)").matches ? document.scrollingElement : dom.scroll;
  scroller.scrollTo({ top: scroller.scrollHeight });
}

// ---------------------------------------------------------------------------
// Mesajlar
// ---------------------------------------------------------------------------

function addUserMessage(text) {
  dom.welcome.hidden = true;
  const avatar = el("span", { className: "avatar avatar--user", attrs: { "aria-hidden": "true" } }, [icon("user")]);
  const node = el("div", { className: "msg msg--user" }, [
    el("p", { className: "bubble" }, [el("span", { className: "visually-hidden", text: "Sorunuz: " }), text]),
    avatar,
  ]);
  dom.thread.append(node);
  scrollThreadToEnd();
}

function assistantShell(children, extraClass = "") {
  return el("div", { className: `msg msg--assistant ${extraClass}`.trim() }, [
    el("span", { className: "avatar", attrs: { "aria-hidden": "true" } }, [icon("cap")]),
    el("div", { className: "msg__body" }, children),
  ]);
}

function addPendingMessage() {
  const node = assistantShell([
    el("div", { className: "msg__card msg__card--pending" }, [
      el("span", { className: "dots", attrs: { "aria-hidden": "true" } }, [el("span"), el("span"), el("span")]),
      el("span", { text: "Belgeler taranıyor ve yanıt hazırlanıyor…" }),
    ]),
  ]);
  dom.thread.append(node);
  scrollThreadToEnd();
  return node;
}

/** Yanıt metnini paragraflara böler; [S1] işaretlerini kaynağa giden düğmelere çevirir. */
function buildAnswerParagraphs(answer, sources, messageId) {
  const knownIds = new Set(sources.map((source) => source.source_id));
  return answer
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const paragraph = el("p");
      let lastIndex = 0;
      for (const match of part.matchAll(CITATION_PATTERN)) {
        if (match.index > lastIndex) paragraph.append(part.slice(lastIndex, match.index));
        const sourceId = match[1];
        if (knownIds.has(sourceId)) {
          const button = el("button", {
            className: "citation",
            text: `[${sourceId}]`,
            attrs: { type: "button", "aria-label": `${sourceId} kaynağını göster`, "aria-controls": "source-list" },
          });
          button.addEventListener("click", () => focusSource(messageId, sourceId));
          paragraph.append(button);
        } else {
          paragraph.append(
            el("span", { className: "citation citation--missing", text: `[${sourceId}]`, attrs: { title: "Bu kaynak yanıtla birlikte gelmedi" } }),
          );
        }
        lastIndex = match.index + match[0].length;
      }
      if (lastIndex < part.length) paragraph.append(part.slice(lastIndex));
      return paragraph;
    });
}

function copyButton(text) {
  const button = el("button", { className: "icon-button", attrs: { type: "button", "aria-label": "Yanıtı kopyala", title: "Yanıtı kopyala" } }, [
    icon("copy"),
  ]);
  const status = el("span", { className: "visually-hidden", attrs: { role: "status" } });
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.replaceChildren(icon("check"));
      status.textContent = "Yanıt kopyalandı";
    } catch {
      status.textContent = "Yanıt kopyalanamadı";
    }
    setTimeout(() => {
      button.replaceChildren(icon("copy"));
      status.textContent = "";
    }, 2000);
  });
  return [button, status];
}

/**
 * AskResponse'u sohbete asistan mesajı olarak ekler ve kaynak panelini günceller.
 * @param {object} response  AskResponse
 * @param {HTMLElement} [replaceNode]  yerini alacağı "yazıyor" mesajı
 */
export function renderAnswer(response, replaceNode) {
  const messageId = `m${++messageSeq}`;
  const sources = Array.isArray(response.sources) ? response.sources : [];
  answers.set(messageId, response);

  const card = el("div", { className: "msg__card answer" });
  if (response.insufficient_context) {
    card.append(
      el("div", { className: "notice notice--neutral" }, [
        icon("info"),
        el("div", {}, [
          el("p", { className: "notice__title", text: "Belgelerde yeterli bilgi bulunamadı" }),
          el("p", { text: "Sistem cevap uydurmak yerine bunu açıkça belirtir. Soruyu farklı ifade etmeyi deneyebilir veya ilgili birime danışabilirsiniz." }),
        ]),
      ]),
    );
  }
  card.append(...buildAnswerParagraphs(response.answer || "", sources, messageId));

  const meta = el("div", { className: "msg__meta" }, [
    ...copyButton(response.answer || ""),
    el("span", { className: "msg__latency", text: `Yanıt süresi ${formatMs(response.metrics?.total_latency_ms)}` }),
  ]);

  const node = assistantShell([card, meta]);
  node.dataset.messageId = messageId;
  if (replaceNode) replaceNode.replaceWith(node);
  else dom.thread.append(node);

  setActiveMessage(messageId);
  scrollThreadToEnd();
  return node;
}

function addErrorMessage(error, question, replaceNode) {
  const box = el("div");
  renderError(box, error, "Yanıt alınamadı");
  const retry = el("button", { className: "button button--secondary button--small", attrs: { type: "button" } }, [
    icon("refresh", "icon--sm"),
    "Tekrar dene",
  ]);
  const node = assistantShell([el("div", { className: "msg__card msg__card--error" }, [box]), el("div", { className: "msg__meta" }, [retry])]);
  retry.addEventListener("click", () => {
    node.remove();
    // Aynı soru tekrar gönderilir; kullanıcı mesajı zaten sohbette olduğu için yeniden eklenmez.
    ask(question, { addUserBubble: false });
  });
  replaceNode.replaceWith(node);
  scrollThreadToEnd();
}

function setActiveMessage(messageId) {
  activeMessageId = messageId;
  dom.thread.querySelectorAll(".msg--assistant").forEach((node) => {
    node.classList.toggle("is-active", node.dataset.messageId === messageId);
  });
  const response = answers.get(messageId);
  renderSources(response?.sources || [], { insufficient: Boolean(response?.insufficient_context) });
}

function focusSource(messageId, sourceId) {
  if (activeMessageId !== messageId) setActiveMessage(messageId);
  const card = document.getElementById(`source-${sourceId}`);
  if (!card) return;
  dom.sourceList.querySelectorAll(".is-highlighted").forEach((node) => node.classList.remove("is-highlighted"));
  card.classList.add("is-highlighted");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.focus({ preventScroll: true });
}

// ---------------------------------------------------------------------------
// Kaynak paneli
// ---------------------------------------------------------------------------

function buildSourceCard(source) {
  const children = [
    el("div", { className: "source-card__head" }, [
      icon("file"),
      el("div", {}, [
        el("p", { className: "source-card__title" }, [
          el("span", { className: "source-card__id", text: source.source_id }),
          source.document_title,
        ]),
        el("span", { className: "page-chip", text: `Sayfa ${source.page_number}` }),
      ]),
    ]),
    el("blockquote", { className: "source-card__excerpt", text: `“${source.excerpt}”` }),
  ];

  const link = externalLink(source.source_url, [icon("external", "icon--sm"), "Belgeyi aç"], "source-link");
  if (link.tagName === "A") {
    link.setAttribute("aria-label", `${source.document_title} belgesini yeni sekmede aç`);
    children.push(el("div", { className: "source-card__footer" }, [link]));
  }

  return el("li", { className: "source-card", attrs: { id: `source-${source.source_id}`, tabindex: "-1" } }, children);
}

/**
 * @param {Array} sources  AskResponse.sources
 * @param {{insufficient?: boolean}} [options]
 */
export function renderSources(sources, { insufficient = false } = {}) {
  dom.sourceList.replaceChildren(...sources.map(buildSourceCard));
  dom.sourcesCount.textContent = sources.length ? `${sources.length} parça` : "";
  dom.sourcesEmpty.hidden = sources.length > 0;
  dom.sourcesNote.hidden = !(insufficient && sources.length);
  dom.sourcesNote.textContent = "Bu parçalar incelendi ancak soruyu yanıtlamak için yeterli bulunmadı.";
}

// ---------------------------------------------------------------------------
// Soru gönderme
// ---------------------------------------------------------------------------

function readQuestion() {
  const question = dom.question.value.trim();
  let message = "";
  if (question.length < QUESTION_MIN) message = `Soru en az ${QUESTION_MIN} karakter olmalıdır.`;
  else if (question.length > QUESTION_MAX) message = `Soru en fazla ${QUESTION_MAX} karakter olabilir.`;
  showQuestionError(message);
  return message ? null : question;
}

function showQuestionError(message) {
  dom.questionError.hidden = !message;
  dom.questionError.replaceChildren(...(message ? [icon("alert", "icon--sm"), message] : []));
  dom.question.setAttribute("aria-invalid", String(Boolean(message)));
}

async function ask(question, { addUserBubble = true } = {}) {
  const token = conversation;
  setChatState("busy");
  if (addUserBubble) addUserMessage(question);
  const pending = addPendingMessage();

  try {
    const response = await askQuestion(question, STUDENT_METHOD);
    if (token !== conversation) return;
    renderAnswer(response, pending);
  } catch (error) {
    if (token !== conversation) return;
    addErrorMessage(error, question, pending);
  } finally {
    if (token === conversation) setChatState("idle");
  }
}

export async function submitQuestion(event) {
  if (event) event.preventDefault();
  if (inFlight) return;
  const question = readQuestion();
  if (!question) {
    dom.question.focus();
    return;
  }
  dom.question.value = "";
  resizeComposer();
  updateCounter();
  await ask(question);
  dom.question.focus();
}

function askPreset(question) {
  if (inFlight) return;
  collapseSidebar();
  showQuestionError("");
  ask(question).then(() => dom.question.focus());
}

function startNewChat() {
  conversation += 1;
  setChatState("empty");
  dom.question.value = "";
  resizeComposer();
  updateCounter();
  showQuestionError("");
  collapseSidebar();
  dom.question.focus();
}

// ---------------------------------------------------------------------------
// Yazma alanı ve mobil menü
// ---------------------------------------------------------------------------

function resizeComposer() {
  dom.question.style.height = "auto";
  dom.question.style.height = `${Math.min(dom.question.scrollHeight, COMPOSER_MAX_HEIGHT)}px`;
}

function updateCounter() {
  const length = dom.question.value.length;
  dom.counter.textContent = length >= COUNTER_FROM ? `${length} / ${QUESTION_MAX}` : "";
  if (dom.question.getAttribute("aria-invalid") === "true" && dom.question.value.trim().length >= QUESTION_MIN) {
    showQuestionError("");
  }
}

function collapseSidebar() {
  dom.sidebarMenu.dataset.collapsed = "true";
  dom.sidebarToggle.setAttribute("aria-expanded", "false");
  dom.sidebarToggle.setAttribute("aria-label", "Menüyü aç");
}

function toggleSidebar() {
  const open = dom.sidebarMenu.dataset.collapsed === "true";
  dom.sidebarMenu.dataset.collapsed = String(!open);
  dom.sidebarToggle.setAttribute("aria-expanded", String(open));
  dom.sidebarToggle.setAttribute("aria-label", open ? "Menüyü kapat" : "Menüyü aç");
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

function init() {
  const byId = (id) => document.getElementById(id);
  Object.assign(dom, {
    form: byId("ask-form"),
    question: byId("question"),
    questionError: byId("question-error"),
    counter: byId("question-counter"),
    submit: byId("ask-submit"),
    scroll: byId("thread-scroll"),
    thread: byId("thread"),
    welcome: byId("welcome"),
    sourceList: byId("source-list"),
    sourcesCount: byId("sources-count"),
    sourcesEmpty: byId("sources-empty"),
    sourcesNote: byId("sources-note"),
    newChat: byId("new-chat"),
    sidebarMenu: byId("sidebar-menu"),
    sidebarToggle: byId("sidebar-toggle"),
  });

  initStatus(document.querySelector("[data-status]"), { quietWhenOk: true });

  dom.form.addEventListener("submit", submitQuestion);
  dom.question.addEventListener("input", () => {
    resizeComposer();
    updateCounter();
  });
  dom.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      dom.form.requestSubmit();
    }
  });

  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => askPreset(button.dataset.question));
  });
  dom.newChat.addEventListener("click", startNewChat);
  dom.sidebarToggle.addEventListener("click", toggleSidebar);
  collapseSidebar();

  setChatState("empty");
}

init();
