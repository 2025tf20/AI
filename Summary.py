from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pprint import pprint

from config import API_KEY_ENV_VAR, MODEL, SYSTEM_PROMPT

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    raise ImportError(
        "The 'langchain' and 'langchain-openai' packages are required. "
        "Install them with `pip install langchain langchain-openai`."
    ) from exc


class Summary:
    """
    LangChain-backed helper for asking questions to an LLM with a configurable prompt.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dotenv_path: Optional[os.PathLike[str] | str] = None,
        default_temperature: float = 0.2,
    ):
        self.model = model or MODEL
        self.api_key = api_key or self._load_api_key(dotenv_path)
        self.default_temperature = default_temperature
        self._client = self._build_client(self.default_temperature)

    def _build_client(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=temperature,
        )

    def _load_api_key(
        self, dotenv_path: Optional[os.PathLike[str] | str] = None
    ) -> str:
        existing = os.getenv(API_KEY_ENV_VAR)
        if existing:
            return existing

        self._load_dotenv(dotenv_path)
        refreshed = os.getenv(API_KEY_ENV_VAR)
        if refreshed:
            return refreshed

        raise RuntimeError(
            f"API key not found. Set {API_KEY_ENV_VAR} in your environment or .env file."
        )

    def _load_dotenv(
        self, dotenv_path: Optional[os.PathLike[str] | str] = None
    ) -> None:
        path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
        if not path.exists():
            return

        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def ask(
        self,
        question: str,
        *,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a question to the LLM with an optional custom system prompt and context.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        system_prompt = prompt or SYSTEM_PROMPT
        messages = [SystemMessage(content=system_prompt)]

        if context:
            messages.append(
                HumanMessage(content=f"Additional context:\n{context.strip()}")
            )

        messages.append(HumanMessage(content=question.strip()))

        # Reuse the default client unless a call-specific temperature is provided.
        client = (
            self._build_client(temperature)
            if temperature is not None and temperature != self.default_temperature
            else self._client
        )
        response = client.invoke(messages)
        pprint(response)
        return response.content.strip()


if __name__ == "__main__":
    sum = Summary()

    ans = sum.ask(
        """
임대차 계약서에 들어가 특약사항으로 추가할 수 있도록
임대한 방에서 토끼를 기르고 싶은에 이를 특약에 추가해줘
        """
    )

    print(ans)