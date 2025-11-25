from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

T = TypeVar("T")


def _timeout_wrapper(func: Callable[[], T], timeout_seconds: float) -> T:
    """Run a function with a timeout, raising TimeoutError if it exceeds the limit."""
    result_container: list[T | Exception] = []
    exception_container: list[Exception] = []

    def target():
        try:
            result_container.append(func())
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Gemini API call exceeded {timeout_seconds}s timeout")
    if exception_container:
        raise exception_container[0]
    if not result_container:
        raise RuntimeError("Function completed but returned no result")
    return result_container[0]


@dataclass
class LLMScore:
    support: float
    temporal: float
    mechanistic: float
    rationale: str


class GeminiVerifier:
    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        temperature: float = 0.2,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str | None = None,
        base_url_env: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        env_name = api_key_env or "GEMINI_API_KEY"
        resolved_key = api_key or os.getenv(env_name)
        if not resolved_key:
            raise RuntimeError(
                f"Gemini API key is missing. Provide --gemini-key or set the {env_name} environment variable."
            )
        resolved_base = base_url or (os.getenv(base_url_env) if base_url_env else None)
        
        # Debug: print configuration (mask API key)
        masked_key = resolved_key[:8] + "..." + resolved_key[-4:] if len(resolved_key) > 12 else "***"
        print(f"[gemini] Config: model={model_name}, base_url={resolved_base}, key={masked_key}")
        
        # Configure Gemini - the library handles the default endpoint automatically
        # Custom endpoints may require different configuration, but let's try standard first
        genai.configure(api_key=resolved_key)
        
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def test_connection(self) -> bool:
        """Test if Gemini API is reachable and credentials are valid."""
        try:
            test_prompt = 'Respond with JSON: {"status":"ok"}'
            response = self.model.generate_content(
                test_prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            payload = self._extract_text(response)
            data = json.loads(payload)
            return data.get("status") == "ok"
        except Exception as e:
            print(f"[gemini] Connection test failed: {type(e).__name__}: {e}")
            return False

    def score(self, node_i: str, node_j: str, contexts: Sequence[str]) -> LLMScore | None:
        if not contexts:
            return None
        prompt = self._build_prompt(node_i, node_j, contexts)
        try:
            start_time = time.time()
            response = _timeout_wrapper(
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                    ),
                ),
                self.timeout_seconds,
            )
            elapsed = time.time() - start_time
            print(f"[gemini] API call completed in {elapsed:.1f}s")
            payload = self._extract_text(response)
            data = json.loads(payload)
            return LLMScore(
                support=self._clamp(data.get("support", 0.0)),
                temporal=self._clamp(data.get("temporal", 0.0)),
                mechanistic=self._clamp(data.get("mechanistic", 0.0)),
                rationale=data.get("rationale", "").strip(),
            )
        except TimeoutError as e:
            print(f"[gemini] TIMEOUT for ({node_i[:30]}.../{node_j[:30]}...): {e}")
            return None
        except Exception as e:
            print(f"[gemini] Score call failed for ({node_i[:30]}.../{node_j[:30]}...): {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _build_prompt(node_i: str, node_j: str, contexts: Sequence[str]) -> str:
        snippets = "\n".join(f"{idx+1}. {ctx}" for idx, ctx in enumerate(contexts))
        return (
            "You verify whether two entities have a causal relationship based on evidence snippets.\n"
            "Return strict JSON with keys support, temporal, mechanistic (floats between 0 and 1) "
            "and rationale (short sentence).\n"
            f"Subject: {node_i}\n"
            f"Object: {node_j}\n"
            "Evidence:\n"
            f"{snippets}\n"
            'Respond only with JSON like {"support":0.8,"temporal":0.6,"mechanistic":0.4,"rationale":"..."}.'
        )

    @staticmethod
    def _extract_text(response: genai.types.GenerateContentResponse) -> str:
        if getattr(response, "text", None):
            return response.text
        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", "")
                if text:
                    chunks.append(text)
        if not chunks:
            raise ValueError("Model response did not contain text content.")
        return "".join(chunks)

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

