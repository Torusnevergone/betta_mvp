from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL","https://api.deepseek.com")
    llm_api_key: str = os.getenv("LLM_API_KEY","")
    llm_model: str = os.getenv("LLM_MODEL","deepseek-chat")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_s", "60"))

def get_settings() -> Settings:
    return Settings()