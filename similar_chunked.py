import orjson
import os

from numpy import dot
from numpy.linalg import norm
from numpy.linalg import norm
from typing import Generator, List, Tuple

from embedding import OpenAIEmbedding, EmbeddingContainer
from max_heap import MaxHeap
from timer import Timer


def load_embeddings(out_dir: str) -> Generator[OpenAIEmbedding, None, None]:
    json_files = [
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".json")
    ]
    for file in json_files:
        with open(file, "rb") as f:
            for embedding in orjson.loads(f.read()):
                yield embedding


def get_cosine_similarity(a: OpenAIEmbedding, b: OpenAIEmbedding) -> float:
    a, b = a["data"][0]["embedding"], b["data"][0]["embedding"]
    return dot(a, b) / (norm(a) * norm(b))


def get_similar_embeddings(
    query_embedding: OpenAIEmbedding,
    db_embeddings: Generator[OpenAIEmbedding, None, None],
    num_results: int,
) -> List[Tuple[OpenAIEmbedding, float]]:
    similar_embeddings = MaxHeap(num_results)
    for embedding in db_embeddings:
        similar_embeddings.push(
            (embedding, get_cosine_similarity(query_embedding, embedding))
        )
    return similar_embeddings.items()


def main():
    out_dir = "out_chunked"

    with Timer():
        query_embedding: OpenAIEmbedding = EmbeddingContainer(
            num_dimensions=1536, max_tokens=8191
        ).to_dict()
        db_embeddings = load_embeddings(out_dir)

    with Timer():
        similar_embeddings = get_similar_embeddings(query_embedding, db_embeddings, 10)
        for score, _ in similar_embeddings:
            print(score)


if __name__ == "__main__":
    main()
