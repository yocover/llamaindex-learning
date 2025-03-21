from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    CustomLLM,
)
from llama_index.core.llms import ChatMessage
from pydantic import Field
import requests
import json
from typing import Iterator


class SiliconflowLLM(CustomLLM):
    """Custom LLM implementation for Siliconflow using llama-index."""

    model_name: str = Field(default="Qwen/Qwen2.5-72B-Instruct")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = Field(default="https://api.siliconflow.cn/v1")
    top_p: float = Field(default=0.7)
    top_k: int = Field(default=50)
    frequency_penalty: float = Field(default=0.5)
    n: int = Field(default=1)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.siliconflow.cn/v1",
        top_p: float = 0.7,
        top_k: int = 50,
        frequency_penalty: float = 0.5,
        n: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the SiliconflowLLM."""
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.n = n

    def _make_request(
        self, messages: List[Dict[str, str]], stream: bool = False
    ) -> Any:
        """Make a request to the Siliconflow API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data,
            stream=stream,
        )
        response.raise_for_status()
        return response

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt."""
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, stream=False)

        try:
            response_json = response.json()
            text = response_json["choices"][0]["message"]["content"]
            return CompletionResponse(text=text)
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing API response: {e}")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream complete the prompt."""
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, stream=True)

        for line in response.iter_lines():
            if not line:
                continue

            try:
                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "):
                    continue

                json_str = line_text[6:]  # Remove "data: " prefix
                if json_str.strip() == "[DONE]":
                    break

                json_data = json.loads(json_str)
                if json_data["choices"][0].get("finish_reason") is not None:
                    break

                delta = json_data["choices"][0].get("delta", {})
                if "content" in delta:
                    text = delta["content"]
                    yield CompletionResponse(text=text)

            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Error parsing streaming response: {e}")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=4096,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> CompletionResponse:
        """Chat with the LLM."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        response = self._make_request(formatted_messages, stream=False)

        try:
            response_json = response.json()
            text = response_json["choices"][0]["message"]["content"]
            return CompletionResponse(text=text)
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing API response: {e}")

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream chat with the LLM."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        response = self._make_request(formatted_messages, stream=True)

        for line in response.iter_lines():
            if not line:
                continue

            try:
                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "):
                    continue

                json_str = line_text[6:]  # Remove "data: " prefix
                if json_str.strip() == "[DONE]":
                    break

                json_data = json.loads(json_str)
                if json_data["choices"][0].get("finish_reason") is not None:
                    break

                delta = json_data["choices"][0].get("delta", {})
                if "content" in delta:
                    text = delta["content"]
                    yield CompletionResponse(text=text)

            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Error parsing streaming response: {e}")
