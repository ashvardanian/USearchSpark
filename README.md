# USearchSpark

Apache Spark pipeline for processing text data, generating embeddings with MLlib, and creating sharded USearch indexes for billion-scale vector search.

## Architecture

1. __Text Processing__: Spark ingests and preprocesses text data
2. __Embedding Generation__: MLlib creates TF-IDF + SVD embeddings  
3. __Vector Sharding__: Partition embeddings into chunks (default: 1B vectors per shard)
4. __USearch Indexing__: Create separate USearch indexes for each shard
5. __Distributed Search__: Query across shards in parallel and merge results

## Usage

Build and run the pipeline:
```bash
sbt assembly
./scripts/run_pipeline.sh /path/to/text/data /output/path
```

Query the indexes:
```bash  
./scripts/run_query.sh /output/path/indexes examples/sample_query_vector.json
```

## Components

- `TextEmbeddingPipeline`: Text preprocessing and TF-IDF/SVD embedding generation
- `VectorShardingService`: Partition vectors using random, hash, or semantic clustering
- `USearchIndexingService`: Create USearch indexes for each shard
- `DistributedQueryService`: Parallel search across sharded indexes
