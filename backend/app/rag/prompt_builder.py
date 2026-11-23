
from backend.app.schemas.retrieval import ChunkRecord

SYSTEM_PROMPT = """Sen CampusLens AI adlı üniversite bilgi asistanısın.
Yalnızca verilen kaynak parçalarındaki bilgilere dayan.
Kaynaklarda olmayan bir bilgiyi ekleme veya tahmin etme.
Her önemli iddianın sonunda [S1] gibi kaynak kimliği kullan.
Bağlam soruyu yanıtlamak için yetersizse bunu açıkça söyle ve
insufficient_context alanını true yap.
Kısa, anlaşılır ve Türkçe yanıt ver.
Yalnızca belirtilen JSON şemasında cevap üret.
"""


def build_prompt(
    question: str,
    chunks: list[ChunkRecord],
) -> tuple[str, dict[str, ChunkRecord]]:
    """Build a source-grounded prompt and source ID mapping."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    source_map = {}
    context_parts = []

    for index, chunk in enumerate(chunks[:5], start=1):
        source_id = f"S{index}"
        source_map[source_id] = chunk

        context_parts.append(
            f"[{source_id}]\n"
            f"Belge: {chunk.document_title}\n"
            f"Sayfa: {chunk.page_number}\n"
            f"Bölüm: {chunk.section_title or 'Belirtilmemiş'}\n"
            f"Metin: {chunk.text}"
        )

    context = "\n\n".join(context_parts)

    user_prompt = (
        f"Kaynaklar:\n{context}\n\n"
        f"Öğrenci sorusu:\n{question}\n\n"
        "Yanıtını yalnızca şu JSON alanlarıyla oluştur:\n"
        "answer, used_source_ids, insufficient_context."
    )

    return user_prompt, source_map
