import orjson
import os
import multiprocessing
import numpy as np

from typing import Generator, List, Tuple
from embedding import OpenAIEmbedding, EmbeddingContainer
from max_heap import MaxHeap
from timer import Timer


def get_cosine_similarity(a: OpenAIEmbedding, b: OpenAIEmbedding) -> float:
    a, b = a["data"][0]["embedding"], b["data"][0]["embedding"]
    a, b = np.array(a), np.array(b)
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return dot_product / norm_product


def calculate_similarity(db_emb, query_embedding):
    # db_emb, query_embedding = t
    # file, query_embedding = file_query_tuple
    # embedding = load_embedding(file)
    similarity = get_cosine_similarity(query_embedding, db_emb)
    return db_emb, similarity


def get_similar_embeddings(
    query_embedding: OpenAIEmbedding,
    db_embeddings: Generator[OpenAIEmbedding, None, None],
    num_results: int,
# ) -> List[Tuple[OpenAIEmbedding, float]]:
) -> MaxHeap:
    similar_embeddings = MaxHeap(num_results)

    # with multiprocessing.Pool() as pool:
    #     similarities = list(
    #         pool.map(
    #             calculate_similarity,
    #             [(db_embedding, query_embedding) for db_embedding in db_embeddings],
    #         )
    #     )
    
    similarities = []
    for db_embedding in db_embeddings:
        similarities.append(calculate_similarity(db_embedding, query_embedding))

    for embedding, similarity in similarities:
        similar_embeddings.push((embedding, similarity))

    return similar_embeddings
    # return similar_embeddings.items()


def load_embeddings(file: str):
    with open(file, "rb") as f:
        chunk_of_embeddings = orjson.loads(f.read())
        # for chunk_of_embeddings in orjson.loads(f.read()):
        for embedding in chunk_of_embeddings:
            yield embedding


# def get_sub_heap(query_embedding, file: str) -> MaxHeap:
# def get_sub_heap(_, file, query_embedding) -> MaxHeap:
def get_sub_heap(t) -> MaxHeap:
    file, query_embedding = t
    n = 10
    loaded_embeddings = load_embeddings(file)
    mh = get_similar_embeddings(query_embedding, loaded_embeddings, n)
    return mh


def main():
    out_dir = "out_chunked"
    # query_embedding: OpenAIEmbedding = EmbeddingContainer(
    #     num_dimensions=1536, max_tokens=8191
    # ).save_to_file("query_embedding.json")
    with open("query_embedding.json", "rb") as f:
        query_embedding = orjson.loads(f.read())
    json_files = [
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".json")
    ]
    num_files = len(json_files)

    with Timer():
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            sub_heaps = pool.map(
                get_sub_heap,
                [(json_file, query_embedding) for json_file in json_files],
            )
        
        master_heap = MaxHeap(capacity=10)
        for sub_heap in sub_heaps:
            print(sub_heap.items())
            for score, emb in sub_heap.items():
                master_heap.push((emb, score))
            print()
        
        top_items = master_heap.items()
        print([item[0] for item in top_items])
    
    # print()
    # print("~~~")
    # print()

    # with Timer():
    #     with multiprocessing.Pool(processes=num_files) as pool:
    #         sub_heaps = pool.map(
    #             get_sub_heap,
    #             [(json_file, query_embedding) for json_file in json_files],
    #         )
        
    #     master_heap = MaxHeap(capacity=10)
    #     for sub_heap in sub_heaps:
    #         for score, emb in sub_heap.items():
    #             master_heap.push((emb, score))
        
    #     top_items = master_heap.items()
    #     print([item[0] for item in top_items])


if __name__ == "__main__":
    main()
