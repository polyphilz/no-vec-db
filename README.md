# no vec db!

## Why are individual files faster than chunked?

From ChatGPT:

"""
Reading and operating on 1,000 JSON files with one value each is faster than reading and operating on 10 JSON files with 100 values each because of the way that modern operating systems cache data in memory.

When you read a file from disk, the operating system will typically read a larger block of data into memory than is strictly necessary to satisfy the read request. This is because it is more efficient to read a large block of data into memory once and then serve subsequent requests from that block than it is to read small blocks of data from disk multiple times.

When you read 1,000 small files, each file is typically smaller than the block size that the operating system uses for disk I/O. As a result, the operating system reads each file into its own separate block of memory, which can be inefficient because it creates a large number of small memory allocations. This can lead to increased overhead in both memory usage and CPU time.

When you read 10 larger files, however, each file is likely to be larger than the block size used by the operating system. As a result, the operating system will read each file into a larger block of memory and can serve subsequent read requests from that block. This can be more efficient because it reduces the number of small memory allocations and reduces the amount of disk I/O required.

In summary, it is generally faster to operate on a smaller number of larger files than a larger number of smaller files because of the way that modern operating systems cache data in memory.
"""

Questions to verify:

1. Is it true that each JSON file is smaller than the block size that the OS uses for disk I/O?

## Misc

- Using multiprocessing works greatly! ~10 seconds to ~4.5 seconds

## DBs

- Qdrant
- Chroma
- Pinecone
- Weaviate
- Milvus
- pgvector on any Postgres provider
- Vespa
- Vald
- Solr
- LangChain in-memory vec stores

- Use this for a good list of the ones LangChain has implemented: https://github.com/hwchase17/langchain/tree/master/langchain/vectorstores

These are not meant to replace trad DBs; simply for vector-first indexing.

Benchmarks:

- https://farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-i/
- https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-ii/
- https://gradientflow.com/the-vector-database-index/

Things to read on vec dbs:

- https://frankzliu.com/blog/a-gentle-introduction-to-vector-databases

## Other proj

- https://github.com/spullara/bfes (rust impl)
