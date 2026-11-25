// Admin genel bakış ekranı (admin.html)
// Belge sayıları, yöntem karşılaştırma grafiği, sistem sağlığı, son belgeler.

import { initAdminShell } from "./admin-shell.js";
import {
  MAX_UPLOAD_MB,
  METHOD_LABELS,
  MOCK_MODE,
  RETRIEVAL_METHODS,
  getEvaluationResults,
  getHealth,
  listDocuments,
} from "./api.js";
import { el, formatDate, formatDecimal, formatMs, formatNumber, formatPercent, icon, renderError } from "./ui.js";

const STATUS_LABELS = {
  uploaded: "Yüklendi",
  processing: "İndeksleniyor",
  indexed: "İndekslendi",
  failed: "Başarısız",
};

const SPLIT_LABELS = { validation: "doğrulama", test: "test" };

const HEALTH_COMPONENTS = [
  { key: "backend", name: "Backend", desc: "API servisleri (soru, belge, değerlendirme)", iconName: "server" },
  { key: "database", name: "Veritabanı", desc: "Belge kayıtları, sorular ve sonuçlar", iconName: "database" },
  { key: "qdrant", name: "Vector store (Qdrant)", desc: "Vektör indeksleri ve anlamsal arama", iconName: "layers" },
];

const HEALTH_STATES = {
  ok: { cls: "pill--ok", label: "Çalışıyor" },
  mock: { cls: "pill--mock", label: "Mock" },
  not_checked: { cls: "", label: "Kontrol edilmiyor" },
  unavailable: { cls: "pill--down", label: "Hazır değil" },
  collection_missing: { cls: "pill--degraded", label: "İndeks yok" },
  down: { cls: "pill--down", label: "Ulaşılamıyor" },
};

const $ = (id) => document.getElementById(id);

function statusPill(status, label) {
  const name = { indexed: "check-circle", failed: "alert", ok: "check-circle", down: "alert" }[status];
  const children = name ? [icon(name)] : [el("span", { className: "pill__dot", attrs: { "aria-hidden": "true" } })];
  children.push(label);
  return el("span", { className: `pill pill--${status}` }, children);
}

// ---------------------------------------------------------------------------
// Belgeler: sayılar ve son yüklenenler
// ---------------------------------------------------------------------------

async function loadDocuments() {
  try {
    const { documents = [] } = (await listDocuments()) || {};
    const indexed = documents.filter((doc) => doc.status === "indexed").length;
    const pending = documents.filter((doc) => doc.status === "uploaded" || doc.status === "processing").length;
    const failed = documents.filter((doc) => doc.status === "failed").length;
    const chunks = documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);

    $("stat-documents").textContent = formatNumber(documents.length);
    $("stat-documents-sub").textContent =
      [pending && `${pending} işleniyor`, failed && `${failed} başarısız`].filter(Boolean).join(" · ") || "Yüklü PDF belgeleri";
    $("stat-indexed").textContent = formatNumber(indexed);
    $("stat-indexed-sub").textContent = documents.length
      ? `${formatPercent(indexed / documents.length)} (${indexed} / ${documents.length})`
      : "Henüz belge yok";
    $("stat-chunks").textContent = formatNumber(chunks);

    renderRecentDocuments(documents);
  } catch (error) {
    ["stat-documents", "stat-indexed", "stat-chunks"].forEach((id) => ($(id).textContent = "—"));
    $("stat-documents-sub").textContent = "Belge bilgisi alınamadı";
    const box = $("recent-state");
    box.className = "";
    renderError(box, error, "Belgeler alınamadı");
  }
}

function renderRecentDocuments(documents) {
  const state = $("recent-state");
  const recent = documents
    .slice()
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);

  if (!recent.length) {
    state.className = "state";
    state.replaceChildren(
      el("p", { text: "Henüz belge yüklenmedi." }),
      el("a", { className: "link-arrow", text: "İlk belgeyi yükle", attrs: { href: "documents.html#upload" } }),
    );
    return;
  }

  state.hidden = true;
  $("recent-wrap").hidden = false;
  $("recent-body").replaceChildren(
    ...recent.map((doc) =>
      el("tr", {}, [
        el("td", { attrs: { "data-label": "Belge adı" } }, [
          el("div", { className: "cell-file" }, [
            icon("file"),
            el("div", {}, [
              el("div", { className: "cell-file__title", text: doc.title }),
              el("div", { className: "cell-file__meta", text: doc.file_name }),
            ]),
          ]),
        ]),
        el("td", { text: formatDate(doc.created_at), attrs: { "data-label": "Yüklenme" } }),
        el("td", { className: "num", text: formatNumber(doc.chunk_count ?? 0), attrs: { "data-label": "Parça" } }),
        el("td", { attrs: { "data-label": "Durum" } }, [statusPill(doc.status, STATUS_LABELS[doc.status] || doc.status)]),
      ]),
    ),
  );
}

// ---------------------------------------------------------------------------
// Yöntem karşılaştırma grafiği (tek seri, tek renk; tablo görünümü evaluation.html'de)
// ---------------------------------------------------------------------------

function latestCompletedPerMethod(results) {
  // Backend'in işaretlediği örnek (is_mock) sonuçlar gerçek ölçüm olmadığı için hiç gösterilmez.
  const completed = results.filter((result) => !result.is_mock && result.status === "completed" && result.metrics);
  if (!completed.length) return { split: null, rows: new Map() };

  // En son koşunun setini kullan; aynı setteki her yöntemin en son sonucunu al.
  completed.sort((a, b) => new Date(b.started_at) - new Date(a.started_at));
  const split = completed[0].split;
  const rows = new Map();
  for (const result of completed) {
    if (result.split === split && !rows.has(result.method)) rows.set(result.method, result);
  }
  return { split, rows };
}

function renderChart(results) {
  const container = $("method-chart");
  const { split, rows } = latestCompletedPerMethod(results);

  if (!rows.size) {
    $("chart-subtitle").textContent = "Gerçek bir değerlendirme koşusu tamamlandığında burada görünür.";
    container.replaceChildren(
      el("div", { className: "state" }, [
        el("p", { className: "state__title", text: "Henüz değerlendirme çalıştırılmadı" }),
        el("a", { className: "link-arrow", text: "Değerlendirme ekranına git", attrs: { href: "evaluation.html" } }),
      ]),
    );
    return;
  }

  $("chart-subtitle").textContent =
    `Son ${SPLIT_LABELS[split] || split} koşusunda Source Recall@5 (yüksek daha iyi)`;

  const chart = el("div", {
    className: "barchart",
    attrs: { role: "list", "aria-label": "Yöntemlere göre Source Recall@5" },
  });

  for (const method of RETRIEVAL_METHODS) {
    const result = rows.get(method);
    const label = METHOD_LABELS[method];
    const value = result?.metrics?.recall_at_5;
    chart.append(el("span", { className: "barchart__label", text: label, attrs: { "aria-hidden": "true" } }));

    const track = el("div", { className: "barchart__track" });
    if (typeof value !== "number") {
      track.append(
        el("div", { className: "barchart__row", attrs: { role: "listitem" } }, [
          el("span", { className: "barchart__value muted", text: "Koşulmadı" }),
        ]),
      );
      chart.append(track);
      continue;
    }

    const metrics = result.metrics;
    const bar = el("span", { className: "barchart__bar" });
    bar.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    const tooltip = el("span", { className: "barchart__tooltip", attrs: { "aria-hidden": "true" } }, [
      el("strong", { text: label }),
      el("br"),
      `Recall@5: ${formatPercent(value)} · MRR@5: ${formatDecimal(metrics.mrr_at_5, 2)}`,
      el("br"),
      `Hallucination: ${formatPercent(metrics.hallucination_rate)} · Ort. süre: ${formatMs(metrics.average_latency_ms)}`,
    ]);
    track.append(
      el(
        "div",
        {
          className: "barchart__row",
          attrs: {
            role: "listitem",
            tabindex: "0",
            "aria-label": `${label}: Recall@5 ${formatPercent(value)}, MRR@5 ${formatDecimal(metrics.mrr_at_5, 2)}, hallucination ${formatPercent(metrics.hallucination_rate)}`,
          },
        },
        [bar, el("span", { className: "barchart__value", text: formatDecimal(value, 3) }), tooltip],
      ),
    );
    chart.append(track);
  }

  // Eksen: 0 … 1
  chart.append(
    el("span"),
    el("div", { className: "barchart__axis", attrs: { "aria-hidden": "true" } }, [
      ...[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => el("span", { text: formatDecimal(tick, 1) })),
    ]),
  );

  container.replaceChildren(chart, el("p", { className: "barchart__caption", text: "Source Recall@5 (↑)" }));
}

async function loadEvaluation() {
  try {
    const { results = [] } = (await getEvaluationResults()) || {};
    renderChart(results);
  } catch (error) {
    const box = el("div");
    renderError(box, error, "Değerlendirme sonuçları alınamadı");
    $("method-chart").replaceChildren(box);
  }
}

// ---------------------------------------------------------------------------
// Sistem sağlığı
// ---------------------------------------------------------------------------

function renderHealth(health, latencyMs, error) {
  const list = $("health-list");
  list.replaceChildren(
    ...HEALTH_COMPONENTS.map((component) => {
      let key = error ? "down" : health?.[component.key] || "not_checked";
      if (MOCK_MODE) key = "mock";
      const state = HEALTH_STATES[key] || { cls: "", label: key };
      const side = el("div", { className: "health-item__side" }, [
        el("span", { className: `pill ${state.cls}` }, [
          el("span", { className: "pill__dot", attrs: { "aria-hidden": "true" } }),
          state.label,
        ]),
      ]);
      if (component.key === "backend" && !error && typeof latencyMs === "number") {
        side.append(el("span", { className: "muted", text: `Yanıt süresi: ${formatMs(latencyMs)}` }));
      }
      return el("li", { className: "health-item" }, [
        icon(component.iconName),
        el("div", {}, [
          el("p", { className: "health-item__name", text: component.name }),
          el("p", { className: "health-item__desc", text: component.desc }),
        ]),
        side,
      ]);
    }),
  );

  const summary = $("stat-health");
  if (MOCK_MODE) summary.replaceChildren(statusPill("mock", "Mock veri"));
  else if (error) summary.replaceChildren(statusPill("down", "Ulaşılamıyor"));
  else {
    // /health: status "ok" → tüm bileşenler hazır, "degraded" → backend açık ama bir bileşen hazır değil.
    const allOk =
      health?.status === "ok" || (!health?.status && HEALTH_COMPONENTS.every((c) => health?.[c.key] === "ok"));
    summary.replaceChildren(allOk ? statusPill("ok", "Sağlıklı") : statusPill("degraded", "Kısmen hazır"));
  }
}

async function loadHealth() {
  const started = performance.now();
  try {
    const health = await getHealth();
    renderHealth(health, performance.now() - started, null);
  } catch (error) {
    renderHealth(null, null, error);
  }
}

// ---------------------------------------------------------------------------

function init() {
  if (!initAdminShell()) return;

  document.querySelectorAll("[data-max-upload]").forEach((node) => (node.textContent = String(MAX_UPLOAD_MB)));
  $("health-refresh").addEventListener("click", loadHealth);

  loadDocuments();
  loadEvaluation();
  loadHealth();
}

init();
