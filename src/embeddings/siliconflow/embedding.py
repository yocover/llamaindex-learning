import logging

import aiohttp
import requests
from typing import List, Optional
from pydantic import Field

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager

logger = logging.getLogger(__name__)

"""SiliconFlow API URL."""
URL = "https://api.siliconflow.cn/v1/embeddings"
"""Maximum batch size for embedding requests."""
MAX_BATCH_SIZE = 1024
"""User agent"""
USER_AGENT = "llama-index-embeddings-siliconflow-custom"


class SiliconflowEmbedding(BaseEmbedding):
    """
    A wrapper class for accessing embedding models available via the SiliconFlow API. This class allows for easy integration
    of SiliconFlow embeddings into your projects, supporting both synchronous and asynchronous retrieval of text embeddings.

    Args:
        model_name (str): Identifier for the model to be used for embeddings.
        normalize (bool): Flag to normalize embeddings post retrieval. Defaults to False.
        api_key (str): SiliconFlow API key.

    Examples:
        >>> model = SiliconflowEmbedding(model_name="your-model-name", api_key="your-api-key")
        >>> print(model.get_text_embedding("Hello, world!"))
        [0.1, 0.2, 0.3, ...]
    """

    model_name: str = Field(description="The model name to use for embeddings")
    _normalize: bool = PrivateAttr()
    _api_key: str = PrivateAttr()
    _query_prefix: str = PrivateAttr()
    _text_prefix: str = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        api_key: str,
        normalize: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        query_prefix: str = "",
        text_prefix: str = "",
        embed_batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        """
        Init params.
        """
        super().__init__(
            model_name=model_name,
            callback_manager=callback_manager,
            embed_batch_size=embed_batch_size
        )

        self._normalize = normalize
        self._api_key = api_key
        self._query_prefix = query_prefix
        self._text_prefix = text_prefix

    def _post(self, data: List[str]) -> List[List[float]]:
        """
        Sends a POST request to the SiliconFlow Inference API with the given data and returns the API response.
        Input data is chunked into batches to avoid exceeding the maximum batch size (1024).

        Args:
            data (List[str]): A list of strings to be embedded.

        Returns:
            dict: A dictionary containing embeddings from the API.
        """
        chunked_data = _chunk(data, self.embed_batch_size)
        embeddings = []
        for chunk in chunked_data:
            for chuck_input in chunk:
                resp = requests.post(
                    URL,
                    json={
                        "model": self.model_name,
                        "input": chuck_input,
                    },
                    headers=self._get_headers(),
                )
                resp.raise_for_status()
                response = resp.json()
                for item in response["data"]:
                    embeddings.append(item["embedding"])
        return embeddings

    async def _apost(self, data: List[str]) -> List[List[float]]:
        """
        Sends a POST request to the SiliconFlow Inference API with the given data and returns the API response.
        Input data is chunked into batches to avoid exceeding the maximum batch size (1024).

        Args:
            data (List[str]): A list of strings to be embedded.
        Output:
            List[float]: A list of embeddings from the API.

        """
        chunked_data = _chunk(data, self.embed_batch_size)
        embeddings = []
        for chunk in chunked_data:
            for chuck_input in chunk:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        URL,
                        json={
                            "model": self.model_name,
                            "input": chuck_input,
                        },
                        headers=self._get_headers(),
                    ) as resp:
                        response = await resp.json()
                        for item in response["data"]:
                            embeddings.append(item["embedding"])
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get query embedding.
        """
        return self._post(self._add_query_prefix([query]))[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Async get query embedding.
        """
        response = await self._apost(self._add_query_prefix([query]))
        return response[0]

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """
        Get query embeddings.
        """
        return self._post(self._add_query_prefix(queries))

    async def _aget_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """
        Async get query embeddings.
        """
        return await self._apost(self._add_query_prefix(queries))

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding.
        """
        return self._post(self._add_text_prefix([text]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Async get text embedding.
        """
        response = await self._apost(self._add_text_prefix([text]))
        return response[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get text embedding.
        """
        return self._post(self._add_text_prefix(texts))

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Async get text embeddings.
        """
        return await self._apost(self._add_text_prefix(texts))

    def _add_query_prefix(self, queries: List[str]) -> List[str]:
        """
        Add query prefix to queries.
        """
        return (
            [self._query_prefix + query for query in queries]
            if self._query_prefix
            else queries
        )

    def _add_text_prefix(self, texts: List[str]) -> List[str]:
        """
        Add text prefix to texts.
        """
        return (
            [self._text_prefix + text for text in texts] if self._text_prefix else texts
        )

    def _get_headers(self) -> dict:
        """
        Get headers.
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }


def _chunk(items: List[str], batch_size: int = MAX_BATCH_SIZE) -> List[List[str]]:
    """
    Chunk items into batches of size batch_size.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
