import orjson
import os
import shutil

from embedding import EmbeddingContainer


def delete_out_dir(out_dir: str) -> None:
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)


def main():
    out_dir = "out_chunked"

    delete_out_dir(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    num_files = 50
    num_embeddings_per_file = 1_000
    for i in range(num_files):
        container = EmbeddingContainer(num_dimensions=1536, max_tokens=8191)
        embeddings = [container.to_dict() for _ in range(num_embeddings_per_file)]
        with open(f"{out_dir}/embedding-{i}.json", "wb") as f:
            f.write(orjson.dumps(embeddings))


if __name__ == "__main__":
    main()
