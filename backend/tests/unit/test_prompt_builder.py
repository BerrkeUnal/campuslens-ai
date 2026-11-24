from backend.app.rag.prompt_builder import build_prompt
from backend.app.schemas.retrieval import ChunkRecord


def test_build_prompt_with_three_sources():
    chunks = [
        ChunkRecord(
            chunk_id="doc1-p4-c1",
            document_id="doc1",
            document_title="Sinav Yonetmeligi",
            source_url="https://example.edu.tr/sinav.pdf",
            page_number=4,
            section_title="Butunleme Sinavi",
            text=(
                "Basarisiz olan ogrenciler butunleme sinavina "
                "girebilir. Devamsizlik nedeniyle basarisiz "
                "olan ogrenciler butunleme sinavina giremez."
            ),
        ),
        ChunkRecord(
            chunk_id="doc2-p2-c1",
            document_id="doc2",
            document_title="Akademik Takvim",
            source_url="https://example.edu.tr/takvim.pdf",
            page_number=2,
            section_title="Sinav Tarihleri",
            text="Butunleme sinavlari Temmuz ayinda yapilir.",
        ),
        ChunkRecord(
            chunk_id="doc3-p7-c1",
            document_id="doc3",
            document_title="Ders Kayit Kilavuzu",
            source_url="https://example.edu.tr/kayit.pdf",
            page_number=7,
            section_title="Ders Kaydi",
            text="Ogrenciler ders kayitlarini sistemden yapar.",
        ),
    ]

    question = (
        "Devamsizlik nedeniyle kaldigim dersin butunleme sinavina girebilir miyim?"
    )

    prompt, source_map = build_prompt(question, chunks)

    assert len(source_map) == 3
    assert list(source_map.keys()) == ["S1", "S2", "S3"]

    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "[S3]" in prompt

    assert question in prompt

    assert source_map["S1"].page_number == 4
    assert source_map["S2"].document_id == "doc2"
    assert source_map["S3"].chunk_id == "doc3-p7-c1"


def test_empty_question_is_rejected():
    try:
        build_prompt("   ", [])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty question should be rejected")


def test_build_prompt_limits_context_to_five_sources():
    chunks = [
        ChunkRecord(
            chunk_id=f"chunk-{i}",
            document_id="doc-1",
            document_title="Test University Regulation",
            source_url="https://example.edu.tr/test.pdf",
            page_number=i,
            section_title="Test Section",
            text=f"UNIQUE_SOURCE_CONTENT_{i}_END",
        )
        for i in range(1, 21)
    ]

    question = "Butunleme sinavina kimler girebilir?"

    prompt, source_map = build_prompt(question, chunks)

    assert len(source_map) == 5

    assert list(source_map.keys()) == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]

    assert question in prompt

    for i in range(1, 6):
        assert f"[S{i}]" in prompt
        assert chunks[i - 1].text in prompt
        assert source_map[f"S{i}"] == chunks[i - 1]

    for i in range(6, 21):
        assert f"[S{i}]" not in prompt
        assert chunks[i - 1].text not in prompt


def test_build_prompt_does_not_include_api_key(monkeypatch):
    from backend.app.rag.prompt_builder import SYSTEM_PROMPT

    fake_api_key = "TEST_SECRET_API_KEY_DO_NOT_EXPOSE"

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        fake_api_key,
    )

    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_title="Test Regulation",
            source_url="https://example.edu.tr/test.pdf",
            page_number=1,
            text="Public university regulation content.",
        ),
    ]

    question = "What are the exam rules?"

    prompt, source_map = build_prompt(question, chunks)

    assert fake_api_key not in prompt
    assert fake_api_key not in SYSTEM_PROMPT
    assert question in prompt
    assert chunks[0].text in prompt
    assert list(source_map.keys()) == ["S1"]
