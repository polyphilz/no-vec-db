import orjson
import os
import numpy as np
from multiprocessing import Pool
from typing import Generator, List, Tuple
from embedding import OpenAIEmbedding, EmbeddingContainer
from max_heap import MaxHeap
from timer import Timer


def load_embedding(file: str) -> OpenAIEmbedding:
    with open(file, "rb") as f:
        embedding = orjson.loads(f.read())
        return embedding


def load_embeddings(out_dir: str) -> Generator[str, None, None]:
    json_files = [
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".json")
    ]
    for file in json_files:
        yield file


def get_cosine_similarity(a: OpenAIEmbedding, b: OpenAIEmbedding) -> float:
    a, b = a["data"][0]["embedding"], b["data"][0]["embedding"]
    a, b = np.array(a), np.array(b)
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return dot_product / norm_product


def calculate_similarity(file_query_tuple):
    file, query_embedding = file_query_tuple
    embedding = load_embedding(file)
    similarity = get_cosine_similarity(query_embedding, embedding)
    return embedding, similarity


def get_similar_embeddings(
    query_embedding: OpenAIEmbedding,
    db_embeddings: Generator[str, None, None],
    num_results: int,
) -> List[Tuple[OpenAIEmbedding, float]]:
    similar_embeddings = MaxHeap(num_results)

    with Pool() as pool:
        similarities = list(
            pool.map(
                calculate_similarity,
                [(file, query_embedding) for file in db_embeddings],
            )
        )

    for embedding, similarity in similarities:
        similar_embeddings.push((embedding, similarity))

    return similar_embeddings.items()


def main():
    out_dir = "out"

    with Timer():
        # query_embedding: OpenAIEmbedding = EmbeddingContainer(
        #     num_dimensions=1536, max_tokens=8191
        # ).to_dict()
        with open("query_embedding.json", "rb") as f:
            query_embedding = orjson.loads(f.read())
        db_embeddings = load_embeddings(out_dir)

    with Timer():
        similar_embeddings = get_similar_embeddings(query_embedding, db_embeddings, 10)
        for score, _ in similar_embeddings:
            print(score)


if __name__ == "__main__":
    main()
