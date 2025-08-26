# USearchSpark

Vector Search benchmark comparing [USearch](https://github.com/unum-cloud/usearch) HNSW across precisions (`f32`, `f16`, `bf16`, `i8`) vs [Lucene](https://github.com/apache/lucene) HNSW (`f32`) baseline, leveraging Apache [Spark](https://github.com/apache/spark) for distributed indexing and search.

## Quick Start

```bash
gradle run --args="wiki" # run locally on 1 node
gradle run --args="wiki --nodes=4" # split current node into 4
gradle run --args="spacev-1b --k8s=<k8s-cluster-ip>" # run on a remote cluster
```

Benchmarks USearch (`f32`, `f16`, `bf16`, `i8`) against Lucene (`f32`) on Wiki dataset locally, producing output like:

```
Implementation | Precision | Index (ms) | Search (ms) | QPS    | Recall@1 | Recall@10
Lucene         | F32       | 1,250      | 890         | 11,235 | 0.9850   | 0.9950
USearch        | F32       | 980        | 720         | 13,888 | 0.9840   | 0.9945
USearch        | F16       | 1,100      | 650         | 15,384 | 0.9820   | 0.9940
USearch        | BF16      | 1,050      | 680         | 14,705 | 0.9830   | 0.9942
USearch        | I8        | 900        | 480         | 20,833 | 0.9750   | 0.9890
```

## Datasets

| Name                | Size   | Dimensions | Description             |
| ------------------- | ------ | ---------- | ----------------------- |
| `text-to-image`     | 1 GB   | 200        | Yandex Text-to-Image    |
| `wiki`              | 1 GB   | 256        | UForm Wiki embeddings   |
| `creative-captions` | 3 GB   | 256        | UForm Creative Captions |
|                     |        |            |                         |
| `spacev-1b`         | 131 GB | 100        | Microsoft SpaceV-1B     |
| `deep1b`            | 358 GB | 96         | Yandex Deep1B           |
