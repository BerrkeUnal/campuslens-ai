from backend.app.core.config import (
    LLM_MODEL_PRICES_USD_PER_1M_TOKENS,
)


def estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Estimate standard LLM token cost in USD."""

    if input_tokens is not None and input_tokens < 0:
        raise ValueError("Input tokens cannot be negative.")

    if output_tokens is not None and output_tokens < 0:
        raise ValueError("Output tokens cannot be negative.")

    if input_tokens is None or output_tokens is None:
        return None

    prices = LLM_MODEL_PRICES_USD_PER_1M_TOKENS.get(model)

    if prices is None:
        return None

    input_cost = input_tokens * prices["input"]
    output_cost = output_tokens * prices["output"]

    return (input_cost + output_cost) / 1_000_000
