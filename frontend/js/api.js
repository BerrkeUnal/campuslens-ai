// Backend ile tüm iletişim bu dosyadan geçer.
// Sözleşme: apiRequest, askQuestion, uploadDocument, listDocuments,
//           deleteDocument, runEvaluation, getEvaluationResults
// Bu dosyada hiçbir API anahtarı veya sır bulunmaz; LLM çağrıları backend'dedir.

import { mockApi } from "./mock-data.js";

/** API tabanı: tek sabit. Gerekirse yalnızca burayı değiştirin. */
export const API_BASE_URL = "http://127.0.0.1:8000/api";

export const REQUEST_TIMEOUT_MS = 60_000;
// İlk soruda reranker ve LLM modelleri yüklenirken yanıt 60 saniyeyi aşabilir.
export const QUESTION_TIMEOUT_MS = 180_000;
// Yükleme sırasında PDF metne çevrilip indekslenir; büyük belgelerde birkaç dakika sürebilir.
export const UPLOAD_TIMEOUT_MS = 300_000;
const REACHABILITY_TIMEOUT_MS = 4000;
export const MAX_UPLOAD_MB = 20; // backend .env MAX_UPLOAD_MB ile aynı tutulmalı

export const RETRIEVAL_METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"];

export const METHOD_LABELS = {
  bm25: "BM25 (anahtar kelime)",
  dense: "Dense vektör",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + Rerank",
};

// ---------------------------------------------------------------------------
// Mock modu
// ---------------------------------------------------------------------------

const MOCK_STORAGE_KEY = "campuslens.mock";

// Mock modu yalnızca geliştirme ve demo içindir; teslim sürümünde varsayılan olarak KAPALIDIR.
// Yalnızca adrese ?mock=1 eklenerek veya admin üst barındaki düğmeyle, o sekme için açılır
// (sessionStorage). Sekme kapanınca kendiliğinden kapanır; kalıcı olarak açık kalmaz.
function readSessionFlag() {
  try {
    return window.sessionStorage.getItem(MOCK_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function writeSessionFlag(enabled) {
  try {
    if (enabled) window.sessionStorage.setItem(MOCK_STORAGE_KEY, "1");
    else window.sessionStorage.removeItem(MOCK_STORAGE_KEY);
  } catch {
    /* depolama kapalı olabilir; sorun değil */
  }
}

function resolveMockMode() {
  // Önceki sürümler bu ayarı kalıcı (localStorage) tutuyordu; teslimde açık kalmasın diye temizlenir.
  try {
    window.localStorage.removeItem(MOCK_STORAGE_KEY);
  } catch {
    /* yoksay */
  }
  const params = new URLSearchParams(window.location.search);
  if (params.has("mock")) {
    const enabled = params.get("mock") === "1";
    writeSessionFlag(enabled);
    return enabled;
  }
  return readSessionFlag();
}

export const MOCK_MODE = resolveMockMode();

export function setMockMode(enabled) {
  writeSessionFlag(enabled);
  const url = new URL(window.location.href);
  url.searchParams.delete("mock");
  window.location.replace(url.toString());
}

// ---------------------------------------------------------------------------
// Hata modeli
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(message, { status = 0, code = "unknown_error", requestId = null, details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

const STATUS_MESSAGES = {
  400: "İstek geçersiz.",
  404: "İstenen kayıt bulunamadı.",
  413: "Dosya izin verilen boyuttan büyük.",
  415: "Yalnızca PDF dosyaları yüklenebilir.",
  422: "Gönderilen bilgilerden biri geçersiz.",
  429: "Çok fazla istek gönderildi. Lütfen biraz bekleyip tekrar deneyin.",
  500: "Beklenmeyen bir sunucu hatası oluştu.",
  502: "Sunucuya şu anda ulaşılamıyor.",
  503: "Servis geçici olarak kullanılamıyor. Lütfen daha sonra tekrar deneyin.",
  504: "Sunucu zamanında yanıt vermedi.",
};

async function readBody(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// Backend iki biçimde hata döndürebilir:
//  1) ErrorResponse: { error, message, request_id, details }  (uygulama ve doğrulama hataları)
//  2) FastAPI HTTPException: { detail: "..." }                 (ör. 404, 413, 415)
function toApiError(response, body) {
  const status = response.status;
  let message = null;

  // Backend'in Türkçe mesajı varsa her zaman o gösterilir; yoksa duruma göre Türkçe bir metin.
  if (body && typeof body.message === "string" && body.message.trim()) {
    message = body.message.trim();
  } else if (body && typeof body.detail === "string" && body.detail.trim()) {
    message = body.detail.trim();
  }

  if (!message) {
    message = STATUS_MESSAGES[status] || `İstek başarısız oldu (HTTP ${status}).`;
  }

  return new ApiError(message, {
    status,
    code: (body && body.error) || "http_error",
    requestId: (body && body.request_id) || response.headers.get("X-Request-ID"),
    details: (body && Array.isArray(body.details) && body.details) || null,
  });
}

/** Sunucu ayakta mı? Herhangi bir HTTP yanıtı (hata kodu dahil) "ulaşılabilir" sayılır. */
async function isServerReachable() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REACHABILITY_TIMEOUT_MS);
  try {
    await fetch(`${API_BASE_URL}/health`, { signal: controller.signal, cache: "no-store" });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Temel istek fonksiyonu
// ---------------------------------------------------------------------------

/**
 * @param {string} path  "/questions/ask" gibi, API_BASE_URL'e eklenir
 * @param {{method?: string, json?: unknown, formData?: FormData, timeoutMs?: number}} options
 */
export async function apiRequest(path, { method = "GET", json, formData, timeoutMs = REQUEST_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const init = {
    method,
    headers: { Accept: "application/json" },
    signal: controller.signal,
  };

  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (formData) {
    // Content-Type elle yazılmaz; tarayıcı multipart boundary'yi kendisi ekler.
    init.body = formData;
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error) {
    clearTimeout(timer);
    if (error && error.name === "AbortError") {
      throw new ApiError("Sunucu zamanında yanıt vermedi. Lütfen biraz sonra tekrar deneyin.", { code: "timeout" });
    }
    // fetch yalnızca "TypeError" verir: bağlantı hiç kurulamamış da olabilir, sunucu bir hata
    // döndürüp tarayıcı yanıtı okuyamamış da olabilir. "Sunucuya bağlanılamadı" yalnızca
    // sunucuya gerçekten ulaşılamıyorsa gösterilir.
    if (path !== "/health" && (await isServerReachable())) {
      throw new ApiError(
        "Sunucu isteği işlerken bir hata oluştu ve ayrıntı alınamadı. Lütfen tekrar deneyin.",
        { code: "unreadable_response" },
      );
    }
    throw new ApiError(
      "Sunucuya bağlanılamadı. İnternet bağlantınızı veya sunucunun çalıştığını kontrol edin.",
      { code: "network_error" },
    );
  }
  clearTimeout(timer);

  if (response.status === 204) return null;

  const body = await readBody(response);
  if (!response.ok) throw toApiError(response, body);
  return body;
}

// ---------------------------------------------------------------------------
// Uç noktalar
// ---------------------------------------------------------------------------

export function getHealth() {
  if (MOCK_MODE) return mockApi.getHealth();
  return apiRequest("/health", { timeoutMs: 5000 });
}

/** POST /api/questions/ask */
export function askQuestion(question, method = "hybrid_rerank") {
  if (!RETRIEVAL_METHODS.includes(method)) {
    return Promise.reject(new ApiError("Geçersiz arama yöntemi seçildi.", { code: "invalid_method" }));
  }
  if (MOCK_MODE) return mockApi.askQuestion(question, method);
  return apiRequest("/questions/ask", { method: "POST", json: { question, method }, timeoutMs: QUESTION_TIMEOUT_MS });
}

/** POST /api/documents/upload (multipart: file, title, source_url) */
export function uploadDocument({ file, title, sourceUrl }) {
  if (MOCK_MODE) return mockApi.uploadDocument({ file, title, sourceUrl });
  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", title);
  formData.append("source_url", sourceUrl);
  return apiRequest("/documents/upload", { method: "POST", formData, timeoutMs: UPLOAD_TIMEOUT_MS });
}

/** GET /api/documents */
export function listDocuments() {
  if (MOCK_MODE) return mockApi.listDocuments();
  return apiRequest("/documents");
}

/** DELETE /api/documents/{document_id} */
export function deleteDocument(documentId) {
  if (MOCK_MODE) return mockApi.deleteDocument(documentId);
  return apiRequest(`/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
}

/** POST /api/evaluations/run */
export function runEvaluation({ method, split = "test" }) {
  if (MOCK_MODE) return mockApi.runEvaluation({ method, split });
  return apiRequest("/evaluations/run", { method: "POST", json: { method, split }, timeoutMs: 300_000 });
}

/** GET /api/evaluations/results */
export function getEvaluationResults() {
  if (MOCK_MODE) return mockApi.getEvaluationResults();
  return apiRequest("/evaluations/results");
}
