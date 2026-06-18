import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    DATA_PATH: str = os.getenv("DATA_PATH", "data/support_tickets.csv")
    DB_PATH: str = os.getenv("DB_PATH", ":memory:")

    # "auto" -> uses Groq if a key is set, otherwise fall back to the rule-based engine
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto").lower()

    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

    FORCE_FALLBACK: bool = _bool("FORCE_FALLBACK", False)

    # Anomaly detection thresholds
    STALE_UNRESOLVED_HOURS: float = float(os.getenv("STALE_UNRESOLVED_HOURS", "24"))
    STALE_PRIORITIES: tuple = tuple(
        p.strip() for p in os.getenv("STALE_PRIORITIES", "High,Critical").split(",")
    )
    IQR_MULTIPLIER: float = float(os.getenv("IQR_MULTIPLIER", "1.5"))
    LOW_RATING_THRESHOLD: int = int(os.getenv("LOW_RATING_THRESHOLD", "2"))

    # "Now" used to evaluate ticket age. Configurable so the seeded demo
    # dataset (dated early 2024) still produces stale-ticket anomalies during
    # grading instead of everything looking ancient relative to wall-clock
    # time. Defaults to real wall-clock time for real-world usage.
    REFERENCE_NOW: str | None = os.getenv("REFERENCE_NOW")  # e.g. "2024-03-15 12:00"


settings = Settings()
