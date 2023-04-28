import heapq
import json
import multiprocessing
import os
import random
import time
import ujson

from generate_fake_embeddings import generate_fake_values


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed_time = self.end - self.start
        print(f"Elapsed time: {self.elapsed_time:.2f} seconds")


class MaxHeap:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.heap = []

    def push(self, item):
        embedding, similarity_score = item
        heapq.heappush(self.heap, (-similarity_score, embedding))

        if len(self.heap) > self.capacity:
            heapq.heappop(self.heap)

    def pop(self):
        if not self.heap:
            return None

        neg_similarity_score, embedding = heapq.heappop(self.heap)
        return (embedding, -neg_similarity_score)

    def top(self):
        if not self.heap:
            return None

        neg_similarity_score, embedding = self.heap[0]
        return (embedding, -neg_similarity_score)

    def items(self):
        return [
            (-neg_similarity_score, embedding)
            for neg_similarity_score, embedding in self.heap
        ]

    def __len__(self):
        return len(self.heap)


def get_similarity(query_embedding, db_embedding):
    return random.uniform(-1, 1)


def read_file(i, chunk_number, chunk_size, output_dir):
    file_index = chunk_number * chunk_size + i
    file_name = f"{output_dir}fake_value_{file_index:06}.json"
    with open(file_name, "r") as file:
        json_object = ujson.load(file)
    return json_object


def get_fake_embeddings_opt(chunk_size, chunk_number, output_dir):
    chunk = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            read_file,
            [(i, chunk_number, chunk_size, output_dir) for i in range(chunk_size)],
        )

    for result in results:
        chunk.append(result)

    return chunk


def get_fake_embeddings(chunk_size, chunk_number, output_dir):
    chunk = []

    for i in range(chunk_size):
        file_index = chunk_number * chunk_size + i
        file_name = f"{output_dir}fake_value_{file_index:06}.json"
        with open(file_name, "r") as file:
            json_object = json.load(file)
        chunk.append(json_object)

    return chunk


def get_n_most_similar_embeddings(
    query_embedding, chunk_size, n, num_files, output_dir
):
    heap = MaxHeap(capacity=n)

    total_chunks = num_files // chunk_size
    remaining_samples = num_files % chunk_size

    for i in range(total_chunks):
        chunk = get_fake_embeddings_opt(chunk_size, i, output_dir)
        for embedding in chunk:
            similarity = get_similarity(query_embedding, embedding)
            heap.push((embedding, similarity))
        print(f"Chunk #{i + 1} read in and similarities evaluated.")
        del chunk

    if remaining_samples > 0:
        chunk = get_fake_embeddings(chunk_size, total_chunks, output_dir)
        for embedding in chunk:
            similarity = get_similarity(query_embedding, embedding)
            heap.push((embedding, similarity))
        del chunk

    print()
    print(len(heap))


def main():
    with Timer():
        output_dir = "output/"
        num_files = len(os.listdir(output_dir))
        chunk_size = 1000
        n = 10
        fake_query_embedding = generate_fake_values(1)[0]
        similar_embeddings = get_n_most_similar_embeddings(
            fake_query_embedding, chunk_size, n, num_files, output_dir
        )


if __name__ == "__main__":
    main()
