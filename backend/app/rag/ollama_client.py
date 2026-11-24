import json

import httpx
from pydantic import ValidationError

from backend.app.core.config import get_settings
from backend.app.rag.llm_client import (
    LLMClientError,
    LLMGenerationResult,
    LLMOutput,
    LLMUsage,
)
from backend.app.rag.prompt_builder import SYSTEM_PROMPT


class OllamaLLMClient:
    """Generate structured answers with a locally running Ollama model."""

    provider = "ollama"

    def __init__(self) -> None:
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.temperature = settings.llm_temperature

        if not self.base_url:
            raise LLMClientError("Ollama servis adresi yapılandırılmamış.")

        if not self.model.strip():
            raise LLMClientError("Ollama model adı yapılandırılmamış.")

    async def aclose(self) -> None:
        """Ollama requests use short-lived HTTP clients."""
        return

    async def generate(self, user_prompt: str) -> LLMOutput:
        """Generate an answer using the existing common interface."""
        result = await self.generate_with_usage(user_prompt)
        return result.output

    async def generate_with_usage(
        self,
        user_prompt: str,
    ) -> LLMGenerationResult:
        """Generate structured JSON and collect local token usage."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "think": False,
            "format": LLMOutput.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_predict": 400,
            },
        }

        input_tokens = 0
        output_tokens = 0
        usage_available = True

        for attempt in range(2):
            try:
                response = await self._request(payload)
                response.raise_for_status()

            except httpx.TimeoutException as exc:
                raise LLMClientError(
                    "Yerel Ollama modeli zamanında yanıt vermedi."
                ) from exc

            except httpx.ConnectError as exc:
                raise LLMClientError(
                    "Ollama servisine bağlanılamadı. "
                    "Ollama uygulamasının çalıştığını kontrol edin."
                ) from exc

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise LLMClientError(
                        f"Ollama modeli bulunamadı: {self.model}. "
                        f"'ollama pull {self.model}' komutunu çalıştırın."
                    ) from exc

                raise LLMClientError("Ollama servisi isteği işleyemedi.") from exc

            except httpx.RequestError as exc:
                raise LLMClientError("Ollama servisiyle iletişim kurulamadı.") from exc

            try:
                response_data = response.json()

                prompt_eval_count = response_data.get("prompt_eval_count")
                eval_count = response_data.get("eval_count")

                if isinstance(prompt_eval_count, int) and isinstance(eval_count, int):
                    input_tokens += prompt_eval_count
                    output_tokens += eval_count
                else:
                    usage_available = False

                content = response_data["message"]["content"]

                if not isinstance(content, str):
                    raise TypeError("Ollama response content must be a string.")

                output = LLMOutput.model_validate_json(content)

                usage = (
                    LLMUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    if usage_available
                    else None
                )

                return LLMGenerationResult(
                    output=output,
                    usage=usage,
                )

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValidationError,
            ) as exc:
                if attempt == 1:
                    raise LLMClientError(
                        "Ollama iki denemede de geçerli JSON üretemedi."
                    ) from exc

        raise LLMClientError("Ollama yanıtı oluşturulamadı.")

    async def _request(self, payload: dict[str, object]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120.0) as client:
            return await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
