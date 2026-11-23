
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
        "Devamsizlik nedeniyle kaldigim dersin "
        "butunleme sinavina girebilir miyim?"
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
