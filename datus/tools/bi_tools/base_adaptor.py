from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, Field


class QuestionSqlPair(BaseModel):
    chart_id: str = Field(default="", init=True)
    title: str = Field(default="", init=True)
    description: Optional[str] = Field(default="", init=True)
    sql: str = Field(default="", init=True)  # real sql reconstructed sql approximate sql
    origin: str = Field(default="", init=True)  # "native" | "rebuilt" | "approximate"
    extra: Dict[str, Any] = Field(default_factory=dict, init=True)

    def get_question(self):
        question = ""
        if self.title:
            question += self.title
        if self.description:
            if not question:
                question = self.description
            else:
                question += f"  {self.description}"


class BiAdaptorBase(ABC):
    @abstractmethod
    def parse_sql_pair(self, dashboard_url: str) -> Iterable[QuestionSqlPair]:
        raise NotImplementedError


class AuthType(Enum):
    LOGIN = "login"  # username & password
    API_KEY = "api_key"  # api key
