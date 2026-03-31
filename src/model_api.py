import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    backend: str = "openai"   # "dummy" or "openai"
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 128


class BaseModelClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DummyModelClient(BaseModelClient):
    def generate(self, prompt: str) -> str:
        return "DUMMY"


class OpenAIModelClient(BaseModelClient):
    def __init__(self, config: ModelConfig):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다. `pip install openai`를 실행하세요.")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

        self.client = OpenAI(api_key=api_key)
        self.config = config

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.config.model_name,
            input=prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        return response.output_text.strip()


def get_model_client(config: ModelConfig):
    if config.backend == "dummy":
        return DummyModelClient()
    elif config.backend == "openai":
        return OpenAIModelClient(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")