
import json

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.app.core.config import get_settings
from backend.app.rag.prompt_builder import SYSTEM_PROMPT
from backend.app.schemas.retrieval import ChunkRecord


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    used_source_ids: list[str]
    insufficient_context: bool


class LLMClientError(Exception):
    """Raised when LLM generation fails."""


class LLMClient:
    def __init__(self):
        settings = get_settings()

        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key or None,
            timeout=30.0,
            max_retries=0,
        )

    async def generate(self, user_prompt: str) -> LLMOutput:
        """Generate structured JSON, retrying invalid output once."""

        for attempt in range(2):
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=user_prompt,
                    temperature=self.temperature,
                    max_output_tokens=400,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "campuslens_answer",
                            "strict": True,
                            "schema": LLMOutput.model_json_schema(),
                        }
                    },
                )

            except (APITimeoutError, APIConnectionError) as exc:
                raise LLMClientError(
                    "LLM servisine bağlanılamadı veya zaman aşımı oluştu."
                ) from exc

            except RateLimitError as exc:
                raise LLMClientError(
                    "LLM servisinin istek limiti aşıldı."
                ) from exc

            if response.status != "completed":
                raise LLMClientError(
                    "LLM yanıtı tamamlanamadı."
                )

            try:
                data = json.loads(response.output_text)
                return LLMOutput.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 1:
                    raise LLMClientError(
                        "LLM iki denemede de geçerli JSON üretemedi."
                    ) from exc

        raise LLMClientError("LLM yanıtı oluşturulamadı.")


async def answer_with_sources(
    question: str,
    chunks: list["ChunkRecord"],
    llm_client: LLMClient,
) -> LLMOutput:
    """Generate an answer and validate its source citations."""
    from backend.app.rag.prompt_builder import build_prompt
    from backend.app.rag.source_validator import validate_sources

    if not chunks:
        return LLMOutput(
            answer="Bu soruyu yanıtlamak için yeterli kaynak bulunamadı.",
            used_source_ids=[],
            insufficient_context=True,
        )

    prompt, source_map = build_prompt(question, chunks)
    output = await llm_client.generate(prompt)

    return validate_sources(output, source_map)
