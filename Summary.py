from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pprint import pprint

from pydantic import BaseModel, Field

from config import API_KEY_ENV_VAR, MODEL, SYSTEM_PROMPT

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    raise ImportError(
        "The 'langchain' and 'langchain-openai' packages are required. "
        "Install them with `pip install langchain langchain-openai`."
    ) from exc

class Output(BaseModel):
    score: int = Field(..., description="The favorability score for the tenant (1-100).")
    reasoning: list[str] = Field(default_factory=list, description="Detailed reasoning for the score.")
    improvement_suggestions: list[str] = Field(
        default_factory=list, 
        description="Suggestions for improving the lease terms for the tenant."
    )

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
        llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
        ).with_structured_output(Output)

        return llm

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
        response = client.invoke(messages).model_dump()
        #pprint(response)
        return response


if __name__ == "__main__":
    sum = Summary()

    ans = sum.ask(
        """
        임차인은 자전거 1대(자전거 등록증 또는 구매영수증 제출 가능)를 임대인의 서면 승인 하에 실내에 보관할 수 있다.
        1. 보관 위치는 현관 내 지정 구역 또는 임대인과 합의한 별도 장소로 하며, 입주 전 해당 위치를 서면으로 확정한다.
        2. 임차 인은 바닥보호패드 및 자전거 고정장치 등으로 시설 손상을 방지하고, 손상 발생 시 즉시 수리·배상한다.
        3. 자전거 보관으로 인한 악취·오염·통행 방해 등 인접 세대의 정당한 민원이 접수될 경우 임차인은 즉시 시정조치를 취하고 시정 불이행 시 임대인은 보관 금지를 요청할 수 있다.
        """
    )

    print(ans)
    print(type(ans))