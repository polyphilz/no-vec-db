import random
import orjson

from memory_profiler import profile
from typing import List, Dict, Tuple, Union

OpenAIEmbedding = Dict[
    str, Union[str, int, List[Dict[str, Union[int, str, List[float]]]]]
]


class EmbeddingContainer:
    __slots__: Tuple[str, ...] = (
        "_dims",
        "_model_name",
        "_i",
        "_max_tokens",
        "_vec",
        "_tokens",
    )

    def __init__(
        self,
        num_dimensions: int,
        max_tokens: int,
        model_name: str = "text-embedding-ada-002",
        index: int = 0,
    ):
        self._dims: int = num_dimensions
        self._max_tokens: int = max_tokens
        self._model_name: str = model_name
        self._i: int = index
        self._tokens: int = EmbeddingContainer.generate_tokens(self._max_tokens)
        self._vec: List[float] = EmbeddingContainer.generate_embedding_vector(
            self._dims
        )

    @staticmethod
    def generate_embedding_vector(num_dimensions: int) -> List[float]:
        return [random.uniform(-1, 1) for _ in range(num_dimensions)]

    @staticmethod
    def generate_tokens(max_tokens: int) -> int:
        return random.randint(1, max_tokens)

    def to_dict(self) -> OpenAIEmbedding:
        return {
            "data": [{"embedding": self._vec, "index": self._i, "object": "embedding"}],
            "model": self._model_name,
            "object": "list",
            "usage": {"prompt_tokens": self._tokens, "total_tokens": self._tokens},
        }

    def save_to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(self.to_dict()))
