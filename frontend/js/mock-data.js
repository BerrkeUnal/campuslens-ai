// Backend hazır olmadan arayüzü geliştirmek için mock veriler.
// Şekiller backend/app/schemas altındaki Pydantic modelleriyle birebir aynıdır.
// Mock modu: adres çubuğuna ?mock=1 ekleyin ya da üst bardaki "Mock" düğmesini kullanın.
//
// Demo kısayolları (yalnızca mock modunda):
//   - Soru "#yetersiz" içeriyorsa  -> insufficient_context: true yanıtı
//   - Soru "#hata" içeriyorsa      -> 503 servis hatası
//   - Belge başlığı "hata" içeriyorsa -> indeksleme "failed" ile biter

export const mockAskResponse = {
  request_id: "demo-001",
  answer:
    "Devamsızlık nedeniyle başarısız sayılan öğrenci, o dersin bütünleme sınavına giremez. " +
    "Bütünleme sınavına yalnızca devam koşulunu sağlamış ancak dönem sonu sınavında başarısız olan " +
    "ya da sınava girmeyen öğrenciler katılabilir. [S1] Devam koşulu teorik derslerde %70, " +
    "uygulamalı derslerde %80'dir. [S2]",
  insufficient_context: false,
  method: "hybrid_rerank",
  sources: [
    {
      source_id: "S1",
      chunk_id: "regulation-p4-c2",
      document_id: "regulation",
      document_title: "Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği",
      source_url: "https://example.edu.tr/regulation.pdf",
      page_number: 4,
      excerpt:
        "Bütünleme sınavına, devam şartını yerine getirmiş olmak kaydıyla dönem sonu sınavına " +
        "girmeyen veya başarısız olan öğrenciler girebilir. Devamsızlıktan kalan öğrenciler " +
        "bütünleme sınavına alınmaz.",
      retrieval_score: 0.82,
      rerank_score: 0.91,
    },
    {
      source_id: "S2",
      chunk_id: "regulation-p3-c5",
      document_id: "regulation",
      document_title: "Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği",
      source_url: "https://example.edu.tr/regulation.pdf",
      page_number: 3,
      excerpt:
        "Öğrenciler teorik derslerin en az %70'ine, uygulamalı derslerin ise en az %80'ine " +
        "devam etmek zorundadır.",
      retrieval_score: 0.74,
      rerank_score: 0.77,
    },
  ],
  metrics: {
    retrieval_latency_ms: 95,
    generation_latency_ms: 840,
    total_latency_ms: 935,
    estimated_cost_usd: 0.0012,
  },
};

export const mockInsufficientResponse = {
  request_id: "demo-002",
  answer:
    "Yüklenen belgelerde bu soruyu yanıtlamak için yeterli bilgi bulunamadı. " +
    "Lütfen öğrenci işleri birimine danışın.",
  insufficient_context: true,
  method: "hybrid_rerank",
  sources: [
    {
      source_id: "S1",
      chunk_id: "calendar-p1-c1",
      document_id: "calendar",
      document_title: "2026-2027 Akademik Takvim",
      source_url: "https://example.edu.tr/calendar.pdf",
      page_number: 1,
      excerpt: "Güz yarıyılı derslerin başlangıcı: 21 Eylül 2026.",
      retrieval_score: 0.31,
      rerank_score: 0.12,
    },
  ],
  metrics: {
    retrieval_latency_ms: 88,
    generation_latency_ms: 410,
    total_latency_ms: 498,
    estimated_cost_usd: 0.0006,
  },
};

// Backend'deki InMemoryEvaluationService ile aynı mock metrikler.
const mockEvaluationMetrics = {
  bm25: {
    recall_at_5: 0.7,
    mrr_at_5: 0.58,
    answer_correctness: 1.2,
    hallucination_rate: 0.15,
    abstention_accuracy: 0.7,
    average_latency_ms: 150,
    total_cost_usd: 0.01,
  },
  dense: {
    recall_at_5: 0.78,
    mrr_at_5: 0.65,
    answer_correctness: 1.35,
    hallucination_rate: 0.12,
    abstention_accuracy: 0.75,
    average_latency_ms: 190,
    total_cost_usd: 0.01,
  },
  hybrid: {
    recall_at_5: 0.84,
    mrr_at_5: 0.72,
    answer_correctness: 1.5,
    hallucination_rate: 0.1,
    abstention_accuracy: 0.8,
    average_latency_ms: 230,
    total_cost_usd: 0.02,
  },
  hybrid_rerank: {
    recall_at_5: 0.88,
    mrr_at_5: 0.79,
    answer_correctness: 1.65,
    hallucination_rate: 0.08,
    abstention_accuracy: 0.85,
    average_latency_ms: 310,
    total_cost_usd: 0.02,
  },
};

// ---------------------------------------------------------------------------
// Yardımcılar
// ---------------------------------------------------------------------------

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clone = (value) => JSON.parse(JSON.stringify(value));

function mockError(status, code, message) {
  const error = new Error(message);
  error.name = "ApiError";
  error.status = status;
  error.code = code;
  error.requestId = "mock-request";
  return error;
}

function uuid() {
  if (globalThis.crypto && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `mock-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// ---------------------------------------------------------------------------
// Bellek içi durum (sayfa yenilenince sıfırlanır)
// ---------------------------------------------------------------------------

const now = Date.now();

const documents = [
  {
    document_id: uuid(),
    title: "Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği",
    source_url: "https://example.edu.tr/regulation.pdf",
    file_name: "yonetmelik.pdf",
    created_ms: now - 86_400_000,
    page_count: 18,
    chunk_count: 64,
    will_fail: false,
  },
  {
    document_id: uuid(),
    title: "Staj Yönergesi",
    source_url: "https://example.edu.tr/staj.pdf",
    file_name: "staj-yonergesi.pdf",
    created_ms: now - 3_600_000,
    page_count: 7,
    chunk_count: 21,
    will_fail: false,
  },
];

const evaluationResults = [];

// Mock belgelerin durumu yüklenme anından geçen süreye göre ilerler:
// 0-2 sn uploaded, 2-6 sn processing, sonra indexed (veya failed).
function toDocumentResponse(doc) {
  const elapsed = Date.now() - doc.created_ms;
  let status = "indexed";
  if (elapsed < 2000) status = "uploaded";
  else if (elapsed < 6000) status = "processing";
  else if (doc.will_fail) status = "failed";

  const finished = status === "indexed";
  return {
    document_id: doc.document_id,
    title: doc.title,
    source_url: doc.source_url,
    file_name: doc.file_name,
    status,
    page_count: finished ? doc.page_count : null,
    chunk_count: finished ? doc.chunk_count : 0,
    created_at: new Date(doc.created_ms).toISOString(),
    error_message:
      status === "failed" ? "PDF içinden metin çıkarılamadı. Belge taranmış görüntü olabilir." : null,
  };
}

// ---------------------------------------------------------------------------
// api.js ile aynı imzaya sahip mock fonksiyonlar
// ---------------------------------------------------------------------------

export const mockApi = {
  async getHealth() {
    await wait(100);
    return { status: "ok", backend: "mock", database: "mock", qdrant: "mock" };
  },

  async askQuestion(question, method) {
    await wait(700 + Math.random() * 500);

    if (question.includes("#hata")) {
      throw mockError(503, "service_unavailable", "LLM servisine geçici olarak erişilemiyor. Lütfen daha sonra tekrar deneyin.");
    }

    const base = question.includes("#yetersiz") ? mockInsufficientResponse : mockAskResponse;
    const response = clone(base);
    response.request_id = uuid();
    response.method = method;

    // Yöntem farkını göstermek için skorları ve süreleri biraz değiştir.
    const factor = { bm25: 0.78, dense: 0.88, hybrid: 0.94, hybrid_rerank: 1 }[method] ?? 1;
    response.sources = response.sources.map((source) => ({
      ...source,
      retrieval_score: Number((source.retrieval_score * factor).toFixed(3)),
      rerank_score: method === "hybrid_rerank" ? source.rerank_score : null,
    }));
    const retrieval = { bm25: 40, dense: 95, hybrid: 120, hybrid_rerank: 260 }[method] ?? 95;
    response.metrics.retrieval_latency_ms = retrieval;
    response.metrics.total_latency_ms = retrieval + response.metrics.generation_latency_ms;
    return response;
  },

  async uploadDocument({ file, title, sourceUrl }) {
    await wait(900);
    const doc = {
      document_id: uuid(),
      title,
      source_url: sourceUrl,
      file_name: file.name,
      created_ms: Date.now(),
      page_count: Math.max(1, Math.round(file.size / 40_000)),
      chunk_count: Math.max(1, Math.round(file.size / 12_000)),
      will_fail: title.toLocaleLowerCase("tr").includes("hata"),
    };
    documents.unshift(doc);
    return toDocumentResponse(doc);
  },

  async listDocuments() {
    await wait(250);
    const list = documents
      .slice()
      .sort((a, b) => b.created_ms - a.created_ms)
      .map(toDocumentResponse);
    return { documents: list, total: list.length };
  },

  async deleteDocument(documentId) {
    await wait(300);
    const index = documents.findIndex((doc) => doc.document_id === documentId);
    if (index === -1) {
      throw mockError(404, "http_error", "Belge bulunamadı.");
    }
    documents.splice(index, 1);
    return null;
  },

  async runEvaluation({ method, split }) {
    await wait(1200);
    const startedAt = new Date().toISOString();
    const result = {
      run_id: uuid(),
      method,
      split,
      status: "completed",
      metrics: { total_cases: split === "validation" ? 10 : 40, ...mockEvaluationMetrics[method] },
      is_mock: true,
      started_at: startedAt,
      completed_at: new Date().toISOString(),
    };
    evaluationResults.unshift(result);
    const { metrics, completed_at: _completedAt, ...runResponse } = result;
    return runResponse;
  },

  async getEvaluationResults() {
    await wait(250);
    return { results: clone(evaluationResults), total: evaluationResults.length };
  },
};
