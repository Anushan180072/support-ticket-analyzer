import json
import logging
import requests

from app.config import settings

logger = logging.getLogger("ticket_ai.llm")


class LLMUnavailableError(Exception):
    pass


class LLMClient:
    def __init__(self):
        self.provider = self._resolve_provider()
        logger.info(f"LLM provider resolved to: {self.provider}")

    def _resolve_provider(self) -> str:
        if settings.FORCE_FALLBACK:
            return "fallback"
        if settings.LLM_PROVIDER == "groq":
            return "groq"
        if settings.GROQ_API_KEY:
            return "groq"
        return "fallback"

    @property
    def available(self) -> bool:
        return self.provider != "fallback"

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.0) -> str:
        """
        Send a single-turn chat request. Returns the assistant's text content.
        Raises LLMUnavailableError on any failure (timeout, bad response,
        provider not configured) so callers can decide how to degrade.
        """
        if self.provider == "groq":
            return self._chat_groq(system, user, json_mode, temperature)
        raise LLMUnavailableError("No LLM provider is configured or reachable.")

    def _chat_groq(self, system, user, json_mode, temperature) -> str:
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(
                settings.GROQ_BASE_URL, headers=headers, json=payload,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Groq request failed: {e}")
            raise LLMUnavailableError(f"Groq request failed: {e}") from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Groq returned an unexpected response shape: {e}")
            raise LLMUnavailableError("Groq returned an unexpected response.") from e


llm_client = LLMClient()
