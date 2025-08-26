# USearchSpark

Apache Spark pipeline for processing text data, generating embeddings with MLlib, and creating sharded USearch indexes for billion-scale vector search.

## Quick Start

```bash
gradle jar          # Build the project
docker-compose up   # Run via Docker Compose
```

Or run locally with a `SPARK_HOME` preset:

```bash
$SPARK_HOME/bin/spark-submit \
    --class com.ashvardanian.USearchSparkApp \
    --master local[*] \
    --driver-memory 4g \
    --executor-memory 8g \
    build/libs/USearchSpark-0.1.0-SNAPSHOT.jar \
    /path/to/input/data \
    /path/to/output
```

## Components

- `TextEmbeddingPipeline`: Text preprocessing and TF-IDF/SVD embedding generation
- `VectorShardingService`: Partition vectors using random, hash, or semantic clustering
- `USearchIndexingService`: Create USearch indexes for each shard
