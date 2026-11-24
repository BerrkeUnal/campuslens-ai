import json

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ServiceUnavailableError
from backend.app.rag.prompt_builder import SYSTEM_PROMPT
from backend.app.schemas.retrieval import ChunkRecord


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    used_source_ids: list[str]
    insufficient_context: bool


class LLMUsage(BaseModel):
    """Token usage reported by the LLM API."""

    input_tokens: int
    output_tokens: int


class LLMGenerationResult(BaseModel):
    """Structured answer with optional API usage metadata."""

    output: LLMOutput
    usage: LLMUsage | None = None


class LLMClientError(ServiceUnavailableError):
    """Raised when LLM generation fails (HTTP 503)."""

    def __init__(self, detail: str) -> None:
        super().__init__("LLM")
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class LLMClient:
    def __init__(self):
        settings = get_settings()

        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

        api_key = settings.openai_api_key

        if not api_key or not api_key.strip():
            raise LLMClientError("LLM servisi için API anahtarı yapılandırılmamış.")

        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        )

    async def aclose(self) -> None:
        """Close the underlying asynchronous OpenAI client."""
        await self.client.close()

    async def generate(self, user_prompt: str) -> LLMOutput:
        """Generate a structured answer using the existing interface."""
        result = await self.generate_with_usage(user_prompt)
        return result.output

    async def generate_with_usage(
        self,
        user_prompt: str,
    ) -> LLMGenerationResult:
        """Generate structured JSON and collect token usage across attempts."""
        input_tokens = 0
        output_tokens = 0
        usage_available = True

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
                raise LLMClientError("LLM servisinin istek limiti aşıldı.") from exc
            except AuthenticationError as exc:
                raise LLMClientError(
                    "LLM servisi için kimlik doğrulama başarısız oldu."
                ) from exc

            except BadRequestError as exc:
                raise LLMClientError("LLM isteği işlenemedi.") from exc

            except OpenAIError as exc:
                raise LLMClientError(
                    "LLM servisinde beklenmeyen bir hata oluştu."
                ) from exc
            if response.status != "completed":
                raise LLMClientError("LLM yanıtı tamamlanamadı.")

            usage = getattr(response, "usage", None)

            if usage is None:
                usage_available = False
            else:
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens

            try:
                data = json.loads(response.output_text)
                output = LLMOutput.model_validate(data)

                total_usage = (
                    LLMUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    if usage_available
                    else None
                )

                return LLMGenerationResult(
                    output=output,
                    usage=total_usage,
                )

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
    from backend.app.rag.source_validator import (
        SourceValidationError,
        validate_sources,
    )

    if not chunks:
        return LLMOutput(
            answer="Bu soruyu yanıtlamak için yeterli kaynak bulunamadı.",
            used_source_ids=[],
            insufficient_context=True,
        )

    prompt, source_map = build_prompt(question, chunks)
    output = await llm_client.generate(prompt)

    try:
        return validate_sources(output, source_map)
    except SourceValidationError as exc:
        raise LLMClientError(
            "LLM yanıtındaki kaynak doğrulaması başarısız oldu."
        ) from exc
