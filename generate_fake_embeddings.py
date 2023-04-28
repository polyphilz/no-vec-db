"""
Generates fake OpenAI embeddings for testing purposes.

Example OpenAI embeddings API endpoint response:

```
{
  "data": [
    {
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        ...
        -4.547132266452536e-05,
        -0.024047505110502243
      ],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "text-embedding-ada-002",
  "object": "list",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

The "embedding" list will contain 1536 dimensions (i.e. values).
"""
import json
import random
import os


def generate_embeddings(num_samples):
    def generate_randomized_vector(dimensions=1536):
        return [random.uniform(-1, 1) for _ in range(dimensions)]

    def generate_randomized_tokens():
        tokens = random.randint(1, 8191)
        return {"prompt_tokens": tokens, "total_tokens": tokens}

    embeddings = []
    for i in range(num_samples):
        embedding = {
            "data": [
                {
                    "embedding": generate_randomized_vector(),
                    "index": i,
                    "object": "embedding",
                }
            ],
            "model": "text-embedding-ada-002",
            "object": "list",
            "usage": generate_randomized_tokens(),
        }
        embeddings.append(embedding)
    return embeddings


def write_embeddings_to_disk(fake_values, chunk_num, chunk_size, output_dir="output/"):
    os.makedirs(output_dir, exist_ok=True)

    for i, fake_value in enumerate(fake_values):
        file_name = f"{output_dir}fake_value_{(i + (chunk_size * chunk_num)):06}.json"

        with open(file_name, "w") as file:
            json.dump(fake_value, file)


def generate_and_save_chunks(num_samples, chunk_size, output_dir="output/"):
    total_chunks = num_samples // chunk_size
    remaining_samples = num_samples % chunk_size

    for i in range(total_chunks):
        chunk = generate_embeddings(chunk_size)
        write_embeddings_to_disk(chunk, i, chunk_size, output_dir)
        print(f"Chunk #{i + 1} completed.")
        del chunk

    if remaining_samples > 0:
        chunk = generate_embeddings(remaining_samples)
        write_embeddings_to_disk(chunk, total_chunks, chunk_size, output_dir)
        del chunk

    print()
    print(f"{num_samples} fake embeddings generated!")


if __name__ == "__main__":
    num_samples = 10000
    chunk_size = 1000
    generate_and_save_chunks(num_samples, chunk_size)
