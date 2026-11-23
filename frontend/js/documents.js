// Belge yönetimi ekranı (documents.html)
// Sözleşme: submitDocument, refreshDocuments, renderDocumentRow, setUploadState

import { MAX_UPLOAD_MB, deleteDocument, listDocuments, uploadDocument } from "./api.js";
import { el, externalLink, formatDate, initPage, renderError, safeUrl, setButtonBusy } from "./ui.js";

const STATUS_LABELS = {
  uploaded: "Yüklendi",
  processing: "İndeksleniyor",
  indexed: "İndekslendi",
  failed: "Başarısız",
};
const PENDING_STATUSES = new Set(["uploaded", "processing"]);

const POLL_INTERVAL_MS = 3000;
const POLL_MAX_DURATION_MS = 2 * 60 * 1000;
const DELETE_CONFIRM_MS = 4000;

const dom = {};
let uploading = false;
let refreshing = false;
let pollTimer = null;
let pollStartedAt = 0;

// ---------------------------------------------------------------------------
// Yükleme durumu
// ---------------------------------------------------------------------------

/**
 * @param {"idle"|"uploading"|"success"|"error"} state
 * @param {{message?: string, error?: Error}} [options]
 */
export function setUploadState(state, { message, error } = {}) {
  const box = dom.uploadStatus;
  box.className = "state section-spaced";
  box.hidden = state === "idle";
  setButtonBusy(dom.uploadSubmit, state === "uploading", "Yükleniyor…");

  if (state === "uploading") {
    box.classList.add("state--loading");
    box.replaceChildren(
      el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
      el("span", { text: message || "Dosya yükleniyor…" }),
    );
  } else if (state === "success") {
    box.classList.add("state--success");
    box.replaceChildren(el("p", { text: message }));
  } else if (state === "error") {
    box.classList.add("state--error");
    renderError(box, error || { message }, "Yükleme başarısız");
  }
}

// ---------------------------------------------------------------------------
// Form doğrulama ve gönderim
// ---------------------------------------------------------------------------

function markInvalid(input, invalid) {
  input.setAttribute("aria-invalid", String(invalid));
}

function validateUploadForm() {
  const file = dom.file.files && dom.file.files[0];
  const title = dom.title.value.trim();
  const sourceUrl = dom.sourceUrl.value.trim();
  [dom.file, dom.title, dom.sourceUrl].forEach((input) => markInvalid(input, false));

  if (!file) return { input: dom.file, message: "Lütfen bir PDF dosyası seçin." };
  const isPdfName = file.name.toLowerCase().endsWith(".pdf");
  const isPdfType = !file.type || file.type === "application/pdf";
  if (!isPdfName || !isPdfType) return { input: dom.file, message: "Yalnızca PDF dosyaları yüklenebilir." };
  if (file.size === 0) return { input: dom.file, message: "Seçilen dosya boş." };
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return { input: dom.file, message: `Dosya boyutu en fazla ${MAX_UPLOAD_MB} MB olabilir.` };
  }
  if (title.length < 2 || title.length > 200) {
    return { input: dom.title, message: "Belge başlığı 2 ile 200 karakter arasında olmalıdır." };
  }
  if (sourceUrl.length < 8 || !safeUrl(sourceUrl)) {
    return { input: dom.sourceUrl, message: "Geçerli bir http(s) kaynak bağlantısı girin." };
  }
  return { file, title, sourceUrl };
}

export async function submitDocument(event) {
  if (event) event.preventDefault();
  if (uploading) return;

  const result = validateUploadForm();
  if (result.message) {
    markInvalid(result.input, true);
    result.input.focus();
    setUploadState("error", { message: result.message });
    return;
  }

  uploading = true;
  setUploadState("uploading", { message: `“${result.file.name}” yükleniyor…` });

  try {
    const created = await uploadDocument(result);
    dom.form.reset();
    setUploadState("success", {
      message: `“${created.title}” yüklendi. İndeksleme durumu listede otomatik güncellenecek.`,
    });
    await refreshDocuments();
  } catch (error) {
    setUploadState("error", { error });
  } finally {
    uploading = false;
    setButtonBusy(dom.uploadSubmit, false);
  }
}

// ---------------------------------------------------------------------------
// Liste
// ---------------------------------------------------------------------------

function statusBadge(doc) {
  const status = doc.status in STATUS_LABELS ? doc.status : "uploaded";
  const children = [];
  if (PENDING_STATUSES.has(status)) {
    children.push(el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }));
  }
  children.push(STATUS_LABELS[status]);
  return el("span", { className: `badge badge--${status}` }, children);
}

function deleteButton(doc) {
  const button = el("button", {
    className: "button button--danger button--small",
    text: "Sil",
    attrs: { type: "button", "aria-label": `${doc.title} belgesini sil` },
  });
  let confirmTimer = null;

  button.addEventListener("click", async () => {
    // İlk tıklama onay ister, ikinci tıklama siler.
    if (button.dataset.confirm !== "true") {
      button.dataset.confirm = "true";
      button.textContent = "Emin misiniz?";
      button.setAttribute("aria-label", `${doc.title} silinsin mi? Onaylamak için tekrar basın.`);
      confirmTimer = setTimeout(() => {
        button.dataset.confirm = "false";
        button.textContent = "Sil";
        button.setAttribute("aria-label", `${doc.title} belgesini sil`);
      }, DELETE_CONFIRM_MS);
      return;
    }

    clearTimeout(confirmTimer);
    button.dataset.confirm = "false";
    setButtonBusy(button, true, "Siliniyor…");
    try {
      await deleteDocument(doc.document_id);
      setUploadState("success", { message: `“${doc.title}” ve indeks kayıtları silindi.` });
      await refreshDocuments();
    } catch (error) {
      setButtonBusy(button, false);
      button.textContent = "Sil";
      setUploadState("error", { error });
    }
  });

  return button;
}

/** @returns {HTMLTableRowElement} */
export function renderDocumentRow(doc) {
  const titleCell = el("td", { attrs: { "data-label": "Belge" } }, [
    el("div", { className: "doc-title" }, [externalLink(doc.source_url, doc.title)]),
    el("div", { className: "muted", text: doc.file_name }),
  ]);

  const statusCell = el("td", { attrs: { "data-label": "Durum" } }, [statusBadge(doc)]);
  if (doc.status === "failed" && doc.error_message) {
    statusCell.append(el("p", { className: "doc-error muted", text: doc.error_message }));
  }

  return el("tr", { dataset: { documentId: doc.document_id } }, [
    titleCell,
    statusCell,
    el("td", { className: "num", text: doc.page_count ?? "—", attrs: { "data-label": "Sayfa" } }),
    el("td", { className: "num", text: doc.chunk_count ?? 0, attrs: { "data-label": "Parça" } }),
    el("td", { text: formatDate(doc.created_at), attrs: { "data-label": "Yüklenme" } }),
    el("td", { attrs: { "data-label": "İşlem" } }, [deleteButton(doc)]),
  ]);
}

function showListState(state) {
  dom.listLoading.hidden = state !== "loading";
  dom.listError.hidden = state !== "error";
  dom.listEmpty.hidden = state !== "empty";
  dom.tableWrap.hidden = state !== "list";
}

function schedulePolling(documents) {
  clearTimeout(pollTimer);
  pollTimer = null;
  const hasPending = documents.some((doc) => PENDING_STATUSES.has(doc.status));

  if (!hasPending) {
    pollStartedAt = 0;
    return;
  }
  if (!pollStartedAt) pollStartedAt = Date.now();
  if (Date.now() - pollStartedAt > POLL_MAX_DURATION_MS) {
    dom.updated.textContent = "Otomatik yenileme durdu; durumu görmek için Yenile'ye basın.";
    pollStartedAt = 0;
    return;
  }
  pollTimer = setTimeout(() => refreshDocuments({ silent: true }), POLL_INTERVAL_MS);
}

export async function refreshDocuments({ silent = false } = {}) {
  if (refreshing) return;
  refreshing = true;
  if (!silent) {
    setButtonBusy(dom.refresh, true, "Yenileniyor…");
    if (dom.tableWrap.hidden) showListState("loading");
  }

  try {
    const { documents = [] } = (await listDocuments()) || {};
    dom.body.replaceChildren(...documents.map(renderDocumentRow));
    showListState(documents.length ? "list" : "empty");
    dom.updated.textContent = `Son güncelleme: ${new Date().toLocaleTimeString("tr-TR")}`;
    schedulePolling(documents);
  } catch (error) {
    clearTimeout(pollTimer);
    renderError(dom.listError, error, "Belge listesi alınamadı");
    showListState("error");
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
    form: byId("upload-form"),
    file: byId("file"),
    title: byId("title"),
    sourceUrl: byId("source-url"),
    uploadSubmit: byId("upload-submit"),
    uploadStatus: byId("upload-status"),
    refresh: byId("documents-refresh"),
    updated: byId("documents-updated"),
    listLoading: byId("documents-loading"),
    listError: byId("documents-error"),
    listEmpty: byId("documents-empty"),
    tableWrap: byId("documents-table-wrap"),
    body: byId("documents-body"),
  });

  document.querySelectorAll("[data-max-upload]").forEach((node) => {
    node.textContent = String(MAX_UPLOAD_MB);
  });

  // Başlık boşsa dosya adından öner.
  dom.file.addEventListener("change", () => {
    const file = dom.file.files && dom.file.files[0];
    markInvalid(dom.file, false);
    if (file && !dom.title.value.trim()) {
      dom.title.value = file.name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ").trim();
    }
  });

  dom.form.addEventListener("submit", submitDocument);
  dom.refresh.addEventListener("click", () => {
    pollStartedAt = 0;
    refreshDocuments();
  });

  refreshDocuments();
}

init();
