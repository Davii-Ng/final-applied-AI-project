import os
import time
from typing import Any, Dict, Optional


PING_PROMPT = "Reply with exactly: pong"


def check_gemini_connectivity(
    api_key: Optional[str] = None,
    model: str = "gemini-1.5-flash",
    llm_class: Any = None,
    message_class: Any = None,
) -> Dict[str, Any]:
    """Run a minimal Gemini connectivity check and return a structured status payload."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        return {
            "ok": False,
            "provider": "gemini",
            "model": model,
            "error": "missing GOOGLE_API_KEY",
        }

    try:
        if llm_class is None or message_class is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            llm_class = ChatGoogleGenerativeAI
            message_class = HumanMessage

        llm = llm_class(model=model, google_api_key=key, temperature=0)
        started = time.perf_counter()
        response = llm.invoke([message_class(content=PING_PROMPT)])
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        content = str(getattr(response, "content", response)).strip().lower()
        return {
            "ok": "pong" in content,
            "provider": "gemini",
            "model": model,
            "latency_ms": elapsed_ms,
            "response_preview": content[:120],
        }
    except Exception as exc:  # pragma: no cover - runtime/network/provider failures
        return {
            "ok": False,
            "provider": "gemini",
            "model": model,
            "error": str(exc),
        }
