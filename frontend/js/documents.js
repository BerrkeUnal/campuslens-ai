// Admin belge yönetimi ekranı (documents.html)
// Sözleşme: submitDocument, refreshDocuments, renderDocumentRow, setUploadState

import { initAdminShell } from "./admin-shell.js";
import { MAX_UPLOAD_MB, deleteDocument, listDocuments, uploadDocument } from "./api.js";
import { el, formatBytes, formatDate, formatNumber, icon, renderError, safeUrl, setButtonBusy } from "./ui.js";

const STATUS = {
  uploaded: { label: "Yüklendi", icon: "clock" },
  processing: { label: "İndeksleniyor", icon: null },
  indexed: { label: "İndekslendi", icon: "check-circle" },
  failed: { label: "Başarısız", icon: "alert" },
};
const PENDING_STATUSES = new Set(["uploaded", "processing"]);

const POLL_INTERVAL_MS = 3000;
const POLL_MAX_DURATION_MS = 2 * 60 * 1000;

const dom = {};
let selectedFile = null;
let allDocuments = [];
let uploading = false;
let refreshing = false;
let pollTimer = null;
let pollStartedAt = 0;
let openMenu = null;
let pendingDelete = null;

// ---------------------------------------------------------------------------
// Yükleme durumu ve alan hataları
// ---------------------------------------------------------------------------

/**
 * @param {"idle"|"uploading"|"success"|"error"} state
 * @param {{message?: string, error?: Error}} [options]
 */
export function setUploadState(state, { message, error } = {}) {
  const box = dom.uploadStatus;
  box.className = "section-spaced";
  box.removeAttribute("role");
  box.hidden = state === "idle" || state === "uploading";
  setButtonBusy(dom.uploadSubmit, state === "uploading", "Yükleniyor…");

  if (state === "success") {
    box.classList.add("notice", "notice--success");
    box.replaceChildren(icon("check-circle"), el("p", { text: message }));
  } else if (state === "error") {
    renderError(box, error || { message }, "Yükleme başarısız");
  }
}

function setFieldError(input, errorNode, message) {
  input.setAttribute("aria-invalid", String(Boolean(message)));
  errorNode.hidden = !message;
  errorNode.replaceChildren(...(message ? [icon("alert", "icon--sm"), message] : []));
}

function clearFieldErrors() {
  setFieldError(dom.dropzone, dom.fileError, "");
  setFieldError(dom.title, dom.titleError, "");
  setFieldError(dom.sourceUrl, dom.urlError, "");
}

// ---------------------------------------------------------------------------
// Dosya seçimi (tıklama veya sürükle-bırak)
// ---------------------------------------------------------------------------

function fileProblem(file) {
  if (!file) return "Lütfen bir PDF dosyası seçin.";
  const isPdfName = file.name.toLowerCase().endsWith(".pdf");
  const isPdfType = !file.type || file.type === "application/pdf";
  if (!isPdfName || !isPdfType) return "Yalnızca PDF dosyaları yüklenebilir.";
  if (file.size === 0) return "Seçilen dosya boş.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `Dosya boyutu en fazla ${MAX_UPLOAD_MB} MB olabilir.`;
  return "";
}

function selectFile(file) {
  const problem = fileProblem(file);
  if (problem) {
    selectedFile = null;
    dom.file.value = "";
    showSelectedFile();
    setFieldError(dom.dropzone, dom.fileError, problem);
    return;
  }
  selectedFile = file;
  setFieldError(dom.dropzone, dom.fileError, "");
  if (!dom.title.value.trim()) {
    dom.title.value = file.name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ").trim();
  }
  showSelectedFile();
}

function showSelectedFile() {
  dom.dropzone.hidden = Boolean(selectedFile);
  dom.selected.hidden = !selectedFile;
  if (selectedFile) {
    dom.selectedName.textContent = selectedFile.name;
    dom.selectedSize.textContent = formatBytes(selectedFile.size);
  }
}

function clearSelectedFile() {
  selectedFile = null;
  dom.file.value = "";
  showSelectedFile();
  dom.file.focus();
}

function bindDropzone() {
  const zone = dom.dropzone;
  const stop = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      stop(event);
      zone.classList.add("is-dragover");
    }),
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      stop(event);
      zone.classList.remove("is-dragover");
    }),
  );
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) selectFile(file);
  });
  // Bırakma alanı dışına düşen dosyanın tarayıcıda açılmasını engelle.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
}

// ---------------------------------------------------------------------------
// Form gönderimi
// ---------------------------------------------------------------------------

function validateUploadForm() {
  clearFieldErrors();
  const title = dom.title.value.trim();
  const sourceUrl = dom.sourceUrl.value.trim();
  const errors = [];

  const problem = fileProblem(selectedFile);
  if (problem) errors.push([dom.dropzone, dom.fileError, problem, dom.file]);
  if (title.length < 2 || title.length > 200) {
    errors.push([dom.title, dom.titleError, "Belge başlığı 2 ile 200 karakter arasında olmalıdır.", dom.title]);
  }
  if (sourceUrl.length < 8 || !safeUrl(sourceUrl)) {
    errors.push([dom.sourceUrl, dom.urlError, "Geçerli bir http(s) bağlantısı girin.", dom.sourceUrl]);
  }

  errors.forEach(([input, node, message]) => setFieldError(input, node, message));
  if (errors.length) {
    errors[0][3].focus();
    return null;
  }
  return { file: selectedFile, title, sourceUrl };
}

export async function submitDocument(event) {
  if (event) event.preventDefault();
  if (uploading) return;

  const payload = validateUploadForm();
  if (!payload) {
    setUploadState("idle");
    return;
  }

  uploading = true;
  setUploadState("uploading");
  try {
    const created = await uploadDocument(payload);
    dom.form.reset();
    selectedFile = null;
    showSelectedFile();
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
// Satır menüsü (⋯) ve silme onayı
// ---------------------------------------------------------------------------

function closeMenu({ restoreFocus = false } = {}) {
  if (!openMenu) return;
  const { menu, button } = openMenu;
  menu.remove();
  button.setAttribute("aria-expanded", "false");
  if (restoreFocus) button.focus();
  openMenu = null;
}

function menuItems(menu) {
  return [...menu.querySelectorAll(".menu__item")];
}

function openRowMenu(button, doc) {
  const alreadyOpen = openMenu && openMenu.button === button;
  closeMenu();
  if (alreadyOpen) return;

  const items = [];
  const url = safeUrl(doc.source_url);
  if (url) {
    items.push(
      el("a", { className: "menu__item", attrs: { role: "menuitem", href: url, target: "_blank", rel: "noopener noreferrer" } }, [
        icon("external"),
        "Kaynağı aç",
      ]),
    );
  }
  const deleteItem = el("button", { className: "menu__item menu__item--danger", attrs: { type: "button", role: "menuitem" } }, [
    icon("trash"),
    "Belgeyi sil",
  ]);
  deleteItem.addEventListener("click", () => {
    closeMenu();
    askDelete(doc, button);
  });
  items.push(deleteItem);

  const menu = el("div", { className: "menu", attrs: { role: "menu", "aria-label": `${doc.title} işlemleri` } }, items);
  menu.addEventListener("keydown", (event) => {
    const list = menuItems(menu);
    const index = list.indexOf(document.activeElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      list[(index + 1) % list.length].focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      list[(index - 1 + list.length) % list.length].focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeMenu({ restoreFocus: true });
    } else if (event.key === "Tab") {
      closeMenu();
    }
  });

  document.body.append(menu);
  positionMenu(menu, button);
  button.setAttribute("aria-expanded", "true");
  openMenu = { menu, button };
  menuItems(menu)[0].focus();
}

/** Menü body'ye eklenir ve düğmenin altına (yer yoksa üstüne) sabitlenir. */
function positionMenu(menu, button) {
  const rect = button.getBoundingClientRect();
  const margin = 8;
  const width = menu.offsetWidth;
  const height = menu.offsetHeight;
  const left = Math.min(Math.max(margin, rect.right - width), window.innerWidth - width - margin);
  const below = rect.bottom + 4;
  const top = below + height > window.innerHeight - margin ? Math.max(margin, rect.top - height - 4) : below;
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
}

function askDelete(doc, returnFocusTo) {
  pendingDelete = { doc, returnFocusTo };
  dom.deleteDesc.textContent = `“${doc.title}” ve indeks kayıtları kalıcı olarak silinecek. Bu işlem geri alınamaz.`;
  dom.deleteError.hidden = true;
  dom.deleteDialog.showModal();
  dom.deleteCancel.focus();
}

async function confirmDelete() {
  if (!pendingDelete) return;
  const { doc } = pendingDelete;
  setButtonBusy(dom.deleteConfirm, true, "Siliniyor…");
  dom.deleteCancel.disabled = true;
  try {
    await deleteDocument(doc.document_id);
    pendingDelete = null;
    dom.deleteDialog.close();
    setUploadState("success", { message: `“${doc.title}” ve indeks kayıtları silindi.` });
    await refreshDocuments();
    dom.search.focus();
  } catch (error) {
    dom.deleteError.hidden = false;
    renderError(dom.deleteError, error, "Belge silinemedi");
  } finally {
    setButtonBusy(dom.deleteConfirm, false);
    dom.deleteCancel.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Liste
// ---------------------------------------------------------------------------

function statusPill(doc) {
  const status = STATUS[doc.status] ? doc.status : "uploaded";
  const meta = STATUS[status];
  const visual =
    status === "processing"
      ? el("span", { className: "spinner", attrs: { "aria-hidden": "true" } })
      : icon(meta.icon);
  return el("span", { className: `pill pill--${status}` }, [visual, meta.label]);
}

/** @returns {HTMLTableRowElement} */
export function renderDocumentRow(doc) {
  const info = el("div", {}, [
    el("div", { className: "cell-file__title", text: doc.title }),
    el("div", { className: "cell-file__meta", text: doc.file_name }),
  ]);
  if (doc.status === "failed" && doc.error_message) {
    info.append(el("p", { className: "doc-error", text: doc.error_message }));
  }

  const moreButton = el("button", {
    className: "icon-button",
    attrs: { type: "button", "aria-haspopup": "menu", "aria-expanded": "false", "aria-label": `${doc.title} için işlemler` },
  });
  moreButton.append(icon("more"));
  moreButton.addEventListener("click", (event) => {
    event.stopPropagation();
    openRowMenu(moreButton, doc);
  });

  return el("tr", { dataset: { documentId: doc.document_id } }, [
    el("td", { attrs: { "data-label": "Belge" } }, [
      el("div", { className: "cell-file" }, [el("span", { className: "file-badge", text: "PDF", attrs: { "aria-hidden": "true" } }), info]),
    ]),
    el("td", { attrs: { "data-label": "Durum" } }, [statusPill(doc)]),
    el("td", { className: "num", text: doc.page_count ?? "—", attrs: { "data-label": "Sayfa" } }),
    el("td", { className: "num", text: doc.status === "indexed" ? formatNumber(doc.chunk_count ?? 0) : "—", attrs: { "data-label": "Parça" } }),
    el("td", { className: "nowrap", text: formatDate(doc.created_at), attrs: { "data-label": "Yükleme tarihi" } }),
    el("td", { className: "actions-cell" }, [moreButton]),
  ]);
}

function showListState(state) {
  dom.listLoading.hidden = state !== "loading";
  dom.listError.hidden = state !== "error";
  dom.listEmpty.hidden = state !== "empty";
  dom.listNoMatch.hidden = state !== "nomatch";
  dom.tableWrap.hidden = state !== "list";
}

function matchesFilters(doc) {
  const query = dom.search.value.trim().toLocaleLowerCase("tr");
  const filter = dom.statusFilter.value;
  if (filter === "indexed" && doc.status !== "indexed") return false;
  if (filter === "failed" && doc.status !== "failed") return false;
  if (filter === "pending" && !PENDING_STATUSES.has(doc.status)) return false;
  if (!query) return true;
  return `${doc.title} ${doc.file_name}`.toLocaleLowerCase("tr").includes(query);
}

function renderList() {
  closeMenu();
  dom.total.textContent = `Toplam ${allDocuments.length} belge`;
  if (!allDocuments.length) {
    showListState("empty");
    return;
  }
  const visible = allDocuments.filter(matchesFilters);
  dom.body.replaceChildren(...visible.map(renderDocumentRow));
  showListState(visible.length ? "list" : "nomatch");
}

function schedulePolling() {
  clearTimeout(pollTimer);
  pollTimer = null;
  const hasPending = allDocuments.some((doc) => PENDING_STATUSES.has(doc.status));
  if (!hasPending) {
    pollStartedAt = 0;
    return;
  }
  if (!pollStartedAt) pollStartedAt = Date.now();
  if (Date.now() - pollStartedAt > POLL_MAX_DURATION_MS) {
    dom.updated.textContent = "Otomatik yenileme durdu; durumu görmek için yenile düğmesine basın.";
    pollStartedAt = 0;
    return;
  }
  pollTimer = setTimeout(() => refreshDocuments({ silent: true }), POLL_INTERVAL_MS);
}

export async function refreshDocuments({ silent = false } = {}) {
  if (refreshing) return;
  // Menü açıkken arka plan yenilemesi menüyü kapatmasın.
  if (silent && openMenu) {
    pollTimer = setTimeout(() => refreshDocuments({ silent: true }), POLL_INTERVAL_MS);
    return;
  }
  refreshing = true;
  if (!silent && dom.tableWrap.hidden) showListState("loading");

  try {
    const { documents = [] } = (await listDocuments()) || {};
    allDocuments = documents.slice().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    renderList();
    dom.updated.textContent = `Son güncelleme: ${new Date().toLocaleTimeString("tr-TR")}`;
    schedulePolling();
  } catch (error) {
    clearTimeout(pollTimer);
    renderError(dom.listError, error, "Belge listesi alınamadı");
    showListState("error");
  } finally {
    refreshing = false;
  }
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

function init() {
  if (!initAdminShell()) return;

  const byId = (id) => document.getElementById(id);
  Object.assign(dom, {
    form: byId("upload-form"),
    dropzone: byId("dropzone"),
    file: byId("file"),
    fileError: byId("file-error"),
    selected: byId("selected-file"),
    selectedName: byId("selected-file-name"),
    selectedSize: byId("selected-file-size"),
    selectedRemove: byId("selected-file-remove"),
    title: byId("title"),
    titleError: byId("title-error"),
    sourceUrl: byId("source-url"),
    urlError: byId("source-url-error"),
    uploadSubmit: byId("upload-submit"),
    uploadStatus: byId("upload-status"),
    total: byId("documents-total"),
    refresh: byId("documents-refresh"),
    updated: byId("documents-updated"),
    search: byId("doc-search"),
    statusFilter: byId("status-filter"),
    listLoading: byId("documents-loading"),
    listError: byId("documents-error"),
    listEmpty: byId("documents-empty"),
    listNoMatch: byId("documents-nomatch"),
    tableWrap: byId("documents-table-wrap"),
    body: byId("documents-body"),
    deleteDialog: byId("delete-dialog"),
    deleteDesc: byId("delete-desc"),
    deleteError: byId("delete-error"),
    deleteCancel: byId("delete-cancel"),
    deleteConfirm: byId("delete-confirm"),
  });

  document.querySelectorAll("[data-max-upload]").forEach((node) => (node.textContent = String(MAX_UPLOAD_MB)));

  dom.file.addEventListener("change", () => selectFile(dom.file.files?.[0]));
  dom.selectedRemove.addEventListener("click", clearSelectedFile);
  bindDropzone();
  dom.title.addEventListener("input", () => setFieldError(dom.title, dom.titleError, ""));
  dom.sourceUrl.addEventListener("input", () => setFieldError(dom.sourceUrl, dom.urlError, ""));
  dom.form.addEventListener("submit", submitDocument);

  dom.search.addEventListener("input", renderList);
  dom.statusFilter.addEventListener("change", renderList);
  dom.refresh.addEventListener("click", () => {
    pollStartedAt = 0;
    refreshDocuments();
  });

  document.addEventListener("click", (event) => {
    if (openMenu && !openMenu.menu.contains(event.target)) closeMenu();
  });
  window.addEventListener("resize", () => closeMenu());
  window.addEventListener("scroll", () => closeMenu(), true);

  dom.deleteCancel.addEventListener("click", () => dom.deleteDialog.close());
  dom.deleteConfirm.addEventListener("click", confirmDelete);
  dom.deleteDialog.addEventListener("close", () => {
    if (pendingDelete?.returnFocusTo?.isConnected) pendingDelete.returnFocusTo.focus();
    pendingDelete = null;
  });

  refreshDocuments();
}

init();
