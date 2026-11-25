from unittest.mock import patch

import pytest

from backend.app.rag.reranker import Reranker, RerankerModelError
from backend.app.schemas.retrieval import RetrievalMethod, RetrievedChunk


@pytest.fixture
def candidates():
    return [
        RetrievedChunk(
            chunk_id=f"doc1-p{i}-c1",
            document_id="doc1",
            document_title="Sınav Yönetmeliği",
            source_url="https://example.com/regulation",
            page_number=i,
            section_title=f"Madde {i}",
            text=f"Kaynak metni {i}",
            retrieval_method=RetrievalMethod.HYBRID,
            retrieval_score=1.0 / i,
        )
        for i in range(1, 7)
    ]


@pytest.fixture
def cross_encoder():
    with patch("backend.app.rag.reranker.CrossEncoder") as mock:
        yield mock


def test_reranking_order_and_top_k(candidates, cross_encoder):
    model = cross_encoder.return_value
    model.predict.return_value = [0.2, 0.9, 0.4, 0.8, 0.1, 0.7]

    reranker = Reranker("BAAI/bge-reranker-v2-m3")
    result = reranker.rerank("Sınav kuralları nelerdir?", candidates)

    assert len(result) == 5
    assert [item.page_number for item in result] == [2, 4, 6, 3, 1]
    assert [item.rerank_score for item in result] == [0.9, 0.8, 0.7, 0.4, 0.2]


def test_metadata_and_retrieval_scores_are_preserved(candidates, cross_encoder):
    cross_encoder.return_value.predict.return_value = [0.2, 0.9, 0.4, 0.8, 0.1, 0.7]

    originals = {candidate.chunk_id: candidate.model_dump() for candidate in candidates}

    result = Reranker().rerank("Test question", candidates)

    for item in result:
        assert item.model_dump(exclude={"rerank_score"}) == {
            key: value
            for key, value in originals[item.chunk_id].items()
            if key != "rerank_score"
        }

    assert all(
        candidate.model_dump() == originals[candidate.chunk_id]
        for candidate in candidates
    )


def test_equal_scores_preserve_retrieval_order(candidates, cross_encoder):
    cross_encoder.return_value.predict.return_value = [0.5] * 6

    result = Reranker().rerank("Test question", candidates)

    assert [item.chunk_id for item in result] == [
        item.chunk_id for item in candidates[:5]
    ]


def test_model_is_loaded_only_once(candidates, cross_encoder):
    model = cross_encoder.return_value
    model.predict.return_value = [0.5] * 6

    reranker = Reranker("BAAI/bge-reranker-v2-m3")

    cross_encoder.assert_not_called()

    reranker.rerank("First question", candidates)
    reranker.rerank("Second question", candidates)

    cross_encoder.assert_called_once_with(
        "BAAI/bge-reranker-v2-m3",
        device="cpu",
        local_files_only=True,
    )
    assert model.predict.call_count == 2


def test_question_and_chunk_text_pairs(candidates, cross_encoder):
    model = cross_encoder.return_value
    model.predict.return_value = [0.5] * 6

    question = "Test question"
    Reranker().rerank(question, candidates)

    model.predict.assert_called_once_with(
        [(question, candidate.text) for candidate in candidates],
        show_progress_bar=False,
    )


def test_empty_candidates_do_not_load_model(cross_encoder):
    result = Reranker().rerank("Test question", [])

    assert result == []
    cross_encoder.assert_not_called()


def test_custom_top_k(candidates, cross_encoder):
    cross_encoder.return_value.predict.return_value = [0.2, 0.9, 0.4, 0.8, 0.1, 0.7]

    result = Reranker().rerank("Test question", candidates, top_k=2)

    assert len(result) == 2
    assert [item.page_number for item in result] == [2, 4]


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question, candidates, cross_encoder):
    with pytest.raises(ValueError, match="Question must not be empty"):
        Reranker().rerank(question, candidates)

    cross_encoder.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k, candidates, cross_encoder):
    with pytest.raises(ValueError, match="top_k must be positive"):
        Reranker().rerank("Test question", candidates, top_k)

    cross_encoder.assert_not_called()


def test_unexpected_score_count_is_rejected(candidates, cross_encoder):
    cross_encoder.return_value.predict.return_value = [0.5]

    with pytest.raises(ValueError, match="unexpected score count"):
        Reranker().rerank("Test question", candidates)


@pytest.mark.parametrize(
    "invalid_score",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_scores_are_rejected(invalid_score, candidates, cross_encoder):
    cross_encoder.return_value.predict.return_value = [
        invalid_score,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]

    with pytest.raises(ValueError, match="non-finite score"):
        Reranker().rerank("Test question", candidates)


@pytest.mark.parametrize("top_k", [None, 2])
def test_public_rerank_uses_shared_instance(candidates, top_k):
    from backend.app.rag.reranker import rerank

    expected_top_k = 5 if top_k is None else top_k
    expected_result = candidates[:expected_top_k]

    with patch("backend.app.rag.reranker.get_reranker") as get_shared_reranker:
        shared_reranker = get_shared_reranker.return_value
        shared_reranker.rerank.return_value = expected_result

        if top_k is None:
            result = rerank("Test question", candidates)
        else:
            result = rerank("Test question", candidates, top_k)

        get_shared_reranker.assert_called_once_with()

        shared_reranker.rerank.assert_called_once_with(
            "Test question",
            candidates,
            expected_top_k,
        )

        assert result == expected_result


def test_warm_up_loads_model_and_runs_inference(cross_encoder):
    model = cross_encoder.return_value
    model.predict.return_value = [0.9]

    reranker = Reranker("BAAI/bge-reranker-v2-m3")
    reranker.warm_up()

    cross_encoder.assert_called_once_with(
        "BAAI/bge-reranker-v2-m3",
        device="cpu",
        local_files_only=True,
    )

    model.predict.assert_called_once_with(
        [
            (
                "Öğrenci kayıt işlemleri",
                "Öğrenci kayıtları akademik takvime göre yapılır.",
            )
        ],
        show_progress_bar=False,
    )


def test_rerank_reuses_warmed_model(candidates, cross_encoder):
    model = cross_encoder.return_value
    model.predict.side_effect = [
        [0.9],
        [0.2, 0.9, 0.4, 0.8, 0.1, 0.7],
    ]

    reranker = Reranker()

    reranker.warm_up()
    result = reranker.rerank("Test question", candidates)

    assert cross_encoder.call_count == 1
    assert model.predict.call_count == 2
    assert len(result) == 5
    assert result[0].page_number == 2


def test_model_loading_failure_returns_controlled_error(candidates, cross_encoder):
    cross_encoder.side_effect = OSError("Model missing from cache")

    with pytest.raises(RerankerModelError) as error:
        Reranker().rerank("Test question", candidates)

    assert error.value.status_code == 503
    assert error.value.code == "reranker_unavailable"
    assert "Model missing from cache" in error.value.detail
    assert isinstance(error.value.__cause__, OSError)
    assert "Model missing from cache" not in error.value.message


def test_warm_up_loading_failure_returns_controlled_error(
    cross_encoder,
):
    cross_encoder.side_effect = OSError("Model missing from cache")

    with pytest.raises(RerankerModelError) as error:
        Reranker().warm_up()

    assert error.value.status_code == 503
    assert error.value.code == "reranker_unavailable"
    cross_encoder.return_value.predict.assert_not_called()


def test_inference_failure_returns_controlled_error(candidates, cross_encoder):
    cross_encoder.return_value.predict.side_effect = RuntimeError("Inference failed")

    with pytest.raises(RerankerModelError) as error:
        Reranker().rerank("Test question", candidates)

    assert error.value.status_code == 503
    assert error.value.code == "reranker_unavailable"
    assert isinstance(error.value.__cause__, RuntimeError)


def test_warm_up_inference_failure_returns_controlled_error(
    cross_encoder,
):
    cross_encoder.return_value.predict.side_effect = RuntimeError(
        "Warm-up inference failed"
    )

    with pytest.raises(RerankerModelError) as error:
        Reranker().warm_up()

    assert error.value.status_code == 503
    assert error.value.code == "reranker_unavailable"
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.1, 0.2],
        [float("nan")],
        [float("inf")],
    ],
)
def test_invalid_warm_up_scores_are_rejected(scores, cross_encoder):
    cross_encoder.return_value.predict.return_value = scores

    with pytest.raises(RerankerModelError) as error:
        Reranker().warm_up()

    assert error.value.status_code == 503
    assert error.value.code == "reranker_unavailable"
    assert "Invalid reranker warm-up score" in error.value.detail
