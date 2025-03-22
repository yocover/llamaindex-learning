"""Siliconflow llm api."""

from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    CustomLLM,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import ChatResponse, ChatResponseGen, MessageRole
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from pydantic import Field, ConfigDict
import requests
import json
import aiohttp

from llama_index.core.multi_modal_llms import MultiModalLLM, MultiModalLLMMetadata

def siliconflow_response_to_completion_response(
    response: Any,
) -> CompletionResponse:
    """Convert Siliconflow API response to CompletionResponse."""
    if isinstance(response, requests.Response):
        response_json = response.json()
        text = response_json["choices"][0]["message"]["content"]
        if not text:
            text = ""
        return CompletionResponse(text=text, raw=response_json)
    return CompletionResponse(text="", raw=response)


def siliconflow_response_to_chat_response(
    response: Any,
) -> ChatResponse:
    """Convert Siliconflow API response to ChatResponse."""
    if isinstance(response, requests.Response):
        response_json = response.json()
        message = response_json["choices"][0]["message"]
        content = message["content"]
        if not content:
            content = ""
        return ChatResponse(
            message=ChatMessage(
                role=message["role"],
                content=content,
            ),
            raw=response_json,
        )
    return ChatResponse(message=ChatMessage(), raw=response)


def chat_message_to_siliconflow_messages(
    chat_messages: Sequence[ChatMessage],
) -> List[Dict]:
    """Convert ChatMessage sequence to Siliconflow messages format."""
    return [{"role": msg.role, "content": msg.content} for msg in chat_messages]


DEFAULT_CONTEXT_WINDOW = 1024 * 8


def call_with_messages(
    model: str,
    messages: List[Dict],
    parameters: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs: Any,
) -> Dict:
    """Make a synchronous API call to Siliconflow."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": messages,
    }
    if parameters:
        data.update(parameters)

    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=data,
        stream=parameters.get("stream", False) if parameters else False,
    )
    response.raise_for_status()
    return response


async def acall_with_messages(
    model: str,
    messages: List[Dict],
    parameters: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs: Any,
) -> Dict:
    """Make an asynchronous API call to Siliconflow."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": messages,
    }
    if parameters:
        data.update(parameters)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=data,
        ) as response:
            response.raise_for_status()
            return await response.json()


class SiliconflowLLM(CustomLLM):
    """Siliconflow LLM implementation."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        description="The Siliconflow model to use.",
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for authentication.",
        exclude=True,
    )
    api_base: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="The base URL for the Siliconflow API.",
    )
    top_p: Optional[float] = Field(
        default=0.7,
        description="The top-p value to use for sampling.",
    )
    top_k: Optional[int] = Field(
        default=50,
        description="The top-k value to use for sampling.",
    )
    frequency_penalty: Optional[float] = Field(
        default=0.5,
        description="Penalty for repeated tokens.",
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The size of the context window.",
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        api_key: Optional[str] = None,
        api_base: str = "https://api.siliconflow.cn/v1",
        top_p: float = 0.7,
        top_k: int = 50,
        frequency_penalty: float = 0.5,
        callback_manager: Optional[CallbackManager] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            callback_manager=callback_manager,
            context_window=context_window,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get the metadata for the LLM."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    def _get_parameters(self, stream: bool = False) -> Dict:
        """Get the parameters for the API call."""
        return {
            "stream": stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
        }

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete the prompt."""
        messages = [{"role": "user", "content": prompt}]
        parameters = self._get_parameters(stream=False)
        parameters.update(kwargs)

        response = call_with_messages(
            model=self.model_name,
            messages=messages,
            parameters=parameters,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return siliconflow_response_to_completion_response(response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Stream complete the prompt."""
        messages = [{"role": "user", "content": prompt}]
        parameters = self._get_parameters(stream=True)
        parameters.update(kwargs)

        response = call_with_messages(
            model=self.model_name,
            messages=messages,
            parameters=parameters,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    line_text = line.decode("utf-8")
                    if not line_text.startswith("data: "):
                        continue

                    json_str = line_text[6:]
                    if json_str.strip() == "[DONE]":
                        break

                    json_data = json.loads(json_str)
                    if json_data["choices"][0].get("finish_reason") is not None:
                        break

                    delta = json_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        text = delta["content"]
                        content += text
                        yield CompletionResponse(
                            text=content, delta=text, raw=json_data
                        )

                except (KeyError, json.JSONDecodeError) as e:
                    raise ValueError(f"Error parsing streaming response: {e}")

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the LLM."""
        formatted_messages = chat_message_to_siliconflow_messages(messages)
        parameters = self._get_parameters(stream=False)
        parameters.update(kwargs)

        response = call_with_messages(
            model=self.model_name,
            messages=formatted_messages,
            parameters=parameters,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return siliconflow_response_to_chat_response(response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Stream chat with the LLM."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        parameters = self._get_parameters(stream=True)
        parameters.update(kwargs)

        response = call_with_messages(
            model=self.model_name,
            messages=formatted_messages,
            parameters=parameters,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        def gen() -> ChatResponseGen:
            content = ""
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    line_text = line.decode("utf-8")
                    if not line_text.startswith("data: "):
                        continue

                    json_str = line_text[6:]
                    if json_str.strip() == "[DONE]":
                        break

                    json_data = json.loads(json_str)
                    if json_data["choices"][0].get("finish_reason") is not None:
                        break

                    delta = json_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        text = delta["content"]
                        content += text
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT.value,
                                content=content,
                            ),
                            delta=text,
                            raw=json_data,
                        )

                except (KeyError, json.JSONDecodeError) as e:
                    raise ValueError(f"Error parsing streaming response: {e}")

        return gen()

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async complete the prompt."""
        messages = [{"role": "user", "content": prompt}]
        parameters = self._get_parameters(stream=False)
        parameters.update(kwargs)

        response = await acall_with_messages(
            model=self.model_name,
            messages=messages,
            parameters=parameters,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        text = response["choices"][0]["message"]["content"]
        return CompletionResponse(text=text, raw=response)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat with the LLM."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        parameters = self._get_parameters(stream=False)
        parameters.update(kwargs)

        response = await acall_with_messages(
            model=self.model_name,
            messages=formatted_messages,
            parameters=parameters,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        message = response["choices"][0]["message"]
        return ChatResponse(
            message=ChatMessage(
                role=message["role"],
                content=message["content"],
            ),
            raw=response,
        )
