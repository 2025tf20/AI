from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from pprint import pprint
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from config import API_KEY_ENV_VAR, MODEL, SYSTEM_PROMPT, TEMPLATE, TEMPLATE3

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError as exc:
    raise ImportError(
        "The 'langchain' and 'langchain-openai' packages are required. "
        "Install them with `pip install langchain langchain-openai`."
    ) from exc

class Clause(BaseModel):
    """Contact information for a person."""
    demand: str = Field(description="요구 사항")
    recommended_phrasing: list[str] = Field(
        default_factory=list, description="권장 문구들"
    )
    description: str = Field(description="설명")
    caution: list[str] = Field(default_factory=list, description="주의사항들")

    @field_validator("recommended_phrasing", mode="before")
    @classmethod
    def _normalize_phrasing(cls, v):
        """Strip leading bullet markers like '- ' and ensure list[str]."""
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        cleaned = []
        for item in items:
            text = str(item).strip()
            if text.startswith("- "):
                text = text[2:].lstrip()
            cleaned.append(text)
        return cleaned

class ContactInfo(BaseModel):
    clauses: list[Clause] = Field(
        default_factory=list,  # 비워도 되게 하려면 default_factory, 필수라면 Field(...)
        description="각 요구사항에 대한 추천 특약 조항과 주의사항, 설명",
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
        default_temperature: float = 0.0,
    ):
        self.model = model or MODEL
        self.api_key = api_key or self._load_api_key(dotenv_path)
        self.default_temperature = default_temperature
        self._client = self._build_client(self.default_temperature)
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", TEMPLATE3),           # 사용자 입력을 문자열로 그대로 전달
        ])

    def _build_client(self, temperature: float) -> ChatOpenAI:
        llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=temperature,
        ).with_structured_output(ContactInfo)

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

        # Reuse the default client unless a call-specific temperature is provided.
        client = (
            self._build_client(temperature)
            if temperature is not None and temperature != self.default_temperature
            else self._client
        )

        ask_chain = self.chat_prompt | client

        response = ask_chain.invoke({"scenarios": question}).model_dump()["clauses"]

        return response


if __name__ == "__main__":
    sum = Summary()


    ans = sum.ask(
        """
1. 집주인·중개인·수리기사 방문 ‘긴급상황 제외 24시간 전 문자 통보 + 세입자 동의 후 출입 + 부재중 출입 시 전후 사진 공유’ 규칙
        """
    )
