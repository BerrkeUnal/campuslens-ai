// Admin ana ekranı (admin.html)
// Belge ve değerlendirme durumunun kısa özetini gösterir; asıl işlemler kendi sayfalarındadır.

import { METHOD_LABELS, getEvaluationResults, listDocuments } from "./api.js";
import { formatDate, formatPercent, initPage } from "./ui.js";

async function loadDocumentsSummary(node) {
  try {
    const { documents = [] } = (await listDocuments()) || {};
    if (!documents.length) {
      node.textContent = "Henüz belge yüklenmedi.";
      return;
    }
    const count = (status) => documents.filter((doc) => doc.status === status).length;
    const pending = count("uploaded") + count("processing");
    const parts = [`${documents.length} belge`, `${count("indexed")} indekslendi`];
    if (pending) parts.push(`${pending} işleniyor`);
    if (count("failed")) parts.push(`${count("failed")} başarısız`);
    node.textContent = parts.join(" · ");
  } catch (error) {
    node.textContent = `Özet alınamadı: ${error.message}`;
  }
}

async function loadEvaluationSummary(node) {
  try {
    const { results = [] } = (await getEvaluationResults()) || {};
    const completed = results
      .filter((result) => result.status === "completed" && result.metrics)
      .sort((a, b) => new Date(b.started_at) - new Date(a.started_at));

    if (!completed.length) {
      node.textContent = "Henüz tamamlanmış değerlendirme yok.";
      return;
    }
    const latest = completed[0];
    const label = METHOD_LABELS[latest.method] || latest.method;
    node.textContent =
      `Son koşu: ${label}, Recall@5 ${formatPercent(latest.metrics.recall_at_5)} · ` +
      `${formatDate(latest.completed_at || latest.started_at)}`;
  } catch (error) {
    node.textContent = `Özet alınamadı: ${error.message}`;
  }
}

function init() {
  initPage();
  loadDocumentsSummary(document.getElementById("documents-summary"));
  loadEvaluationSummary(document.getElementById("evaluation-summary"));
}

init();
