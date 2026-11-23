// Sayfalar arasında ortak kullanılan küçük yardımcılar.
// Kural: kullanıcı veya API metni asla innerHTML ile basılmaz; her şey textContent ile.

import { MOCK_MODE, getHealth, setMockMode } from "./api.js";

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
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
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

export function externalLink(href, text) {
  const url = safeUrl(href);
  if (!url) return el("span", { text });
  return el("a", { text, attrs: { href: url, target: "_blank", rel: "noopener noreferrer" } });
}

// ---------------------------------------------------------------------------
// Biçimlendirme
// ---------------------------------------------------------------------------

const numberTr = (digits) =>
  new Intl.NumberFormat("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

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
  return new Intl.DateTimeFormat("tr-TR", { dateStyle: "short", timeStyle: "short" }).format(date);
}

// ---------------------------------------------------------------------------
// Buton meşguliyet durumu (çift gönderimi engeller)
// ---------------------------------------------------------------------------

export function setButtonBusy(button, busy, busyLabel) {
  if (!button) return;
  if (busy) {
    if (!button.dataset.label) button.dataset.label = button.textContent;
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.replaceChildren(el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }), busyLabel || button.dataset.label);
  } else {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    if (button.dataset.label) button.textContent = button.dataset.label;
  }
}

/** Hata kutusu: başlık + mesaj + varsa istek kimliği */
export function renderError(container, error, title = "Bir sorun oluştu") {
  const message = (error && error.message) || "Bilinmeyen bir hata oluştu.";
  const children = [el("h3", { text: title }), el("p", { text: message })];
  if (error && error.requestId) {
    children.push(el("p", { className: "state__detail", text: `İstek kimliği: ${error.requestId}` }));
  }
  container.replaceChildren(...children);
}

// ---------------------------------------------------------------------------
// Ortak üst bar: aktif sayfa, sağlık göstergesi, mock düğmesi
// ---------------------------------------------------------------------------

export function initPage() {
  const health = document.querySelector("[data-health]");
  const healthText = document.querySelector("[data-health-text]");
  const toggle = document.querySelector("[data-mock-toggle]");

  if (toggle) {
    toggle.setAttribute("aria-pressed", String(MOCK_MODE));
    toggle.textContent = MOCK_MODE ? "Mock: açık" : "Mock: kapalı";
    toggle.addEventListener("click", () => setMockMode(!MOCK_MODE));
  }

  if (!health || !healthText) return;

  if (MOCK_MODE) {
    health.dataset.status = "mock";
    healthText.textContent = "Mock veri kullanılıyor";
    return;
  }

  healthText.textContent = "Sunucu kontrol ediliyor…";
  getHealth()
    .then(() => {
      health.dataset.status = "ok";
      healthText.textContent = "Sunucu çalışıyor";
    })
    .catch(() => {
      health.dataset.status = "down";
      healthText.textContent = "Sunucuya ulaşılamıyor";
    });
}
