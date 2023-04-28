import os
import shutil

from embedding import EmbeddingContainer


def delete_out_dir(out_dir: str = "out") -> None:
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)


def main():
    out_dir = "out"

    delete_out_dir(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    n = 50_000
    for i in range(n):
        container = EmbeddingContainer(num_dimensions=1536, max_tokens=8191)
        container.save_to_file(f"{out_dir}/embedding-{i}.json")


if __name__ == "__main__":
    main()
