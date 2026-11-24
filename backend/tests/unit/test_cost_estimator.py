import pytest

from backend.app.rag.cost_estimator import estimate_cost


def test_known_model_cost():
    result = estimate_cost(
        model="gpt-4.1-mini",
        input_tokens=1200,
        output_tokens=80,
    )

    assert result == pytest.approx(0.000608)


def test_retry_usage_cost():
    result = estimate_cost(
        model="gpt-4.1-mini",
        input_tokens=2100,
        output_tokens=120,
    )

    assert result == pytest.approx(0.001032)


def test_unknown_model_returns_none():
    result = estimate_cost(
        model="unknown-model",
        input_tokens=1200,
        output_tokens=80,
    )

    assert result is None


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(None, 80), (1200, None), (None, None)],
)
def test_missing_usage_returns_none(
    input_tokens,
    output_tokens,
):
    result = estimate_cost(
        model="gpt-4.1-mini",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    assert result is None


def test_zero_tokens_returns_zero():
    result = estimate_cost(
        model="gpt-4.1-mini",
        input_tokens=0,
        output_tokens=0,
    )

    assert result == 0.0


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(-1, 80), (1200, -1)],
)
def test_negative_tokens_are_rejected(
    input_tokens,
    output_tokens,
):
    with pytest.raises(ValueError):
        estimate_cost(
            model="gpt-4.1-mini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
