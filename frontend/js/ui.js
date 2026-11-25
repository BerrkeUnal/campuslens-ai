// Sayfalar arasında ortak kullanılan küçük yardımcılar.
// Kural: kullanıcı veya API metni asla innerHTML ile basılmaz; her şey textContent ile.

import { MOCK_MODE, getHealth, setMockMode } from "./api.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const ICON_SPRITE = "img/icons.svg";

/**
 * Güvenli DOM oluşturucu.
 * el("p", { className: "muted", text: "Merhaba", attrs: { id: "x" } }, [child1, child2])
 */
export function el(tag, { className, text, attrs, dataset } = {}, children = []) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === false || value === null || value === undefined) continue;
      node.setAttribute(key, value === true ? "" : String(value));
    }
  }
  if (dataset) Object.assign(node.dataset, dataset);
  for (const child of children) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

/** img/icons.svg içindeki bir ikonu dekoratif SVG olarak döndürür. */
export function icon(name, className = "") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", `icon ${className}`.trim());
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `${ICON_SPRITE}#i-${name}`);
  svg.append(use);
  return svg;
}

/** Yalnızca http(s) bağlantılarına izin ver; javascript: vb. engellenir. */
export function safeUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

export function externalLink(href, children, className) {
  const url = safeUrl(href);
  const content = Array.isArray(children) ? children : [children];
  if (!url) return el("span", { className }, content);
  return el("a", { className, attrs: { href: url, target: "_blank", rel: "noopener noreferrer" } }, content);
}

// ---------------------------------------------------------------------------
// Biçimlendirme
// ---------------------------------------------------------------------------

const numberTr = (digits) =>
  new Intl.NumberFormat("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("tr-TR").format(value);
}

export function formatMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${numberTr(2).format(value / 1000)} sn`;
  return `${Math.round(value)} ms`;
}

export function formatCost(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `$${value.toFixed(4)}`;
}

export function formatScore(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

export function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `%${numberTr(1).format(value * 100)}`;
}

export function formatDecimal(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return numberTr(digits).format(value);
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("tr-TR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function formatBytes(bytes) {
  if (typeof bytes !== "number") return "";
  if (bytes < 1024 * 1024) return `${numberTr(0).format(Math.max(1, bytes / 1024))} KB`;
  return `${numberTr(1).format(bytes / (1024 * 1024))} MB`;
}

// ---------------------------------------------------------------------------
// Buton meşguliyet durumu (çift gönderimi engeller)
// ---------------------------------------------------------------------------

export function setButtonBusy(button, busy, busyLabel) {
  if (!button) return;
  if (busy) {
    if (!button.dataset.busy) {
      button.dataset.busy = "true";
      button._originalChildren = [...button.childNodes];
    }
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.replaceChildren(
      el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
      busyLabel || button.textContent.trim(),
    );
  } else if (button.dataset.busy) {
    delete button.dataset.busy;
    button.disabled = false;
    button.removeAttribute("aria-busy");
    if (button._originalChildren) button.replaceChildren(...button._originalChildren);
  } else {
    button.disabled = false;
  }
}

/** Hata kutusu: ikon + başlık + mesaj + varsa istek kimliği */
export function renderError(container, error, title = "Bir sorun oluştu") {
  const message = (error && error.message) || "Bilinmeyen bir hata oluştu.";
  const text = el("div", {}, [el("p", { className: "notice__title", text: title }), el("p", { text: message })]);
  if (error && error.requestId) {
    text.append(el("p", { className: "notice__detail", text: `İstek kimliği: ${error.requestId}` }));
  }
  container.classList.add("notice", "notice--error");
  container.setAttribute("role", "alert");
  container.replaceChildren(icon("alert"), text);
}

// ---------------------------------------------------------------------------
// Sunucu durumu ve mock düğmesi
// ---------------------------------------------------------------------------

/**
 * Üst bardaki küçük sunucu durumu göstergesi.
 * quietWhenOk: öğrenci ekranında her şey yolundaysa gösterge gizlenir.
 */
export function initStatus(node, { quietWhenOk = false, hideWhenMock = false } = {}) {
  if (!node) return;
  const text = node.querySelector("[data-status-text]");
  const set = (status, label) => {
    node.dataset.status = status;
    if (text) text.textContent = label;
    node.hidden = quietWhenOk && (status === "ok" || status === "unknown");
  };

  if (MOCK_MODE) {
    set("mock", "Mock veri kullanılıyor");
    if (hideWhenMock) node.hidden = true; // admin üst barında mock düğmesi bunu zaten gösterir
    return;
  }
  set("unknown", "Sunucu kontrol ediliyor…");
  getHealth()
    .then((health) => {
      // /health "degraded": backend açık ama veritabanı veya Qdrant hazır değil.
      if (health && health.status === "degraded") set("degraded", "Sunucu kısmen hazır");
      else set("ok", "Sunucu çalışıyor");
    })
    .catch(() => set("down", "Sunucuya ulaşılamıyor"));
}

/**
 * Mock modu açıksa sayfanın en üstüne belirgin bir uyarı bandı ekler; kapalıysa hiçbir şey yapmaz.
 * Ekrandaki verilerin geliştirme amaçlı örnek veri olduğu her sayfada açıkça görünür.
 */
export function initMockBanner() {
  if (!MOCK_MODE || document.querySelector(".mock-banner")) return;
  const button = el("button", { className: "mock-banner__button", text: "Mock modu kapat", attrs: { type: "button" } });
  button.addEventListener("click", () => setMockMode(false));
  const banner = el("div", { className: "mock-banner", attrs: { role: "note", "aria-label": "Geliştirme modu" } }, [
    icon("alert"),
    el("p", { className: "mock-banner__text" }, [
      el("strong", { text: "Geliştirme modu (mock): " }),
      "Bu ekrandaki yanıtlar ve belgeler örnek veridir, gerçek değildir. Backend'e bağlanılmıyor.",
    ]),
    button,
  ]);
  document.body.classList.add("has-mock-banner");
  const skipLink = document.querySelector(".skip-link");
  if (skipLink) skipLink.after(banner);
  else document.body.prepend(banner);
}

export function initMockToggle(button) {
  if (!button) return;
  button.setAttribute("aria-pressed", String(MOCK_MODE));
  button.textContent = MOCK_MODE ? "Mock: açık" : "Mock: kapalı";
  button.addEventListener("click", () => setMockMode(!MOCK_MODE));
}
