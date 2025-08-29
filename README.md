# USearchSpark

Vector Search benchmark comparing [USearch](https://github.com/unum-cloud/usearch) HNSW across precisions (`f32`, `f16`, `bf16`, `i8`) vs [Lucene](https://github.com/apache/lucene) in-memory HNSW (`f32`) baseline, leveraging Apache [Spark](https://github.com/apache/spark) for distributed indexing and search.

## Quick Start

```bash
gradle build --warning-mode all # ensure all dependencies are resolved
gradle run --args="-h" # print supported arguments & available options
gradle run --args="unum-wiki-1m" # index & search `f32` vectors across all cores
gradle run --args="yandex-deep-10m --max-vectors 100000" # limit vectors for testing
gradle run --args="msft-spacev-100m --precision f32,i8" # test specific precisions
```

When running on larger machines, consider overriding JVM settings based on your hardware.
Auto-detect optimal settings for your machine:

```bash
# For 750+ GB machines with custom heap size
JAVA_OPTS="-Xmx512g -Xms64g -XX:ParallelGCThreads=$(nproc)" gradle run --args="msft-spacev-100m"

# For development with limited resources
JAVA_OPTS="-Xmx8g -Xms2g" gradle run --args="unum-wiki-1m --max-vectors 100000"
```

For small test runs comparing the impact of multi-threading you may run:

```bash
JAVA_OPTS="-Xms2g -Xmx8g" gradle run --args="unum-wiki-1m --max-vectors 10000 --queries 10000 --batch-size 100 --threads 1"
JAVA_OPTS="-Xms2g -Xmx8g" gradle run --args="unum-wiki-1m --max-vectors 10000 --queries 10000 --batch-size 100 --threads 8"
```

Benchmarks USearch (`f32`, `f16`, `bf16`, `i8`) against Lucene (`f32`) on Wiki dataset locally, producing clean output like, the following results obtained for the 100M `msft-spacev-100m` subset of Microsoft SpaceV:

```
🚀 PERFORMANCE METRICS
┌──────────────┬──────────────┬──────────────┬──────────────┬─────────────┐
│ Engine       │ Precision    │ IPS          │ QPS          │ Memory      │
├──────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ Lucene       │ F32          │ 13,719       │ 2,601        │ 962 MB      │
│ USearch      │ F32          │ 110,779      │ 123,457      │ 1.3 GB      │
│ USearch      │ F16          │ 133,032      │ 142,857      │ 724 MB      │
│ USearch      │ BF16         │ 133,422      │ 156,250      │ 756 MB      │
│ USearch      │ I8           │ 173,340      │ 200,000      │ 500 MB      │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
🎯 RECALL & NDCG METRICS
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Engine      │ Precision   │ Recall@10   │ NDCG@10     │ Recall@100  │ NDCG@100    │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Lucene      │ F32         │ 91.18%      │ 92.95%      │ 93.17%      │ 92.95%      │
│ USearch     │ F32         │ 98.63%      │ 98.88%      │ 98.67%      │ 98.88%      │
│ USearch     │ F16         │ 97.15%      │ 97.94%      │ 98.63%      │ 97.94%      │
│ USearch     │ BF16        │ 88.89%      │ 92.24%      │ 98.68%      │ 92.24%      │
│ USearch     │ I8          │ 64.25%      │ 72.03%      │ 94.19%      │ 72.03%      │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

> __Hardware__: AWS `m7i.metal-48xl` instances (92 physical cores, 192 threads, 2 sockets).
> __OS__: Linux 6.8.0-1024-aws (Ubuntu 22.04.5 LTS).
> __CPU__: Intel Xeon Platinum 8488C @ 2.4GHz.
> __Memory__: 768 GB RAM total.
> __Java__: OpenJDK 21.0.5 with Java Vector API (`--add-modules=jdk.incubator.vector`).
> __JVM__: 128GB heap (`-Xmx128g`) with ZGC garbage collector (`-XX:+UseZGC`) for sub-10ms pauses.
> __Library versions__: USearch v2.20.4, Lucene v9.12.0.

IPS stands for Insertions Per Second, and QPS is Queries Per Second.
Recall@K is computed as a fraction of search queries, where the known "ground-truth" Top-1 result appeared among the Top-K approximate results.
NDCG@K stands for Normalized [Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) at K, which measures the effectiveness of the search results by considering the position of the relevant documents.

## Datasets

The [BigANN benchmark](https://big-ann-benchmarks.com) is a good starting point if you are searching for extensive collections of high-dimensional vectors.
Still, it rarely considers datasets with more than tens of millions of entries.
In the era of 100+ core CPUs and Petabyte-scale storage in 2U servers, larger datasets are required.

| Dataset                                     | Codename           | DType | NDim | Metric |   Size |  $N$ |
| :------------------------------------------ | :----------------- | ----: | ---: | -----: | -----: | ---: |
| [Unum UForm Creative Captions][unum-cc-3m]  | `unum-cc-3m`       | `f32` |  256 |     IP |   3 GB |  100 |
| [Unum UForm Wiki][unum-wiki-1m]             | `unum-wiki-1m`     | `f32` |  256 |     IP |   1 GB |   10 |
| [Yandex Text-to-Image][yandex-t2i] subset   | `yandex-t2i-1m`    | `f32` |  200 |    Cos |   1 GB |  100 |
| [Yandex Deep10M][yandex-deep] subset        | `yandex-deep-10m`  | `f32` |   96 |     L2 |   4 GB |  100 |
| [Microsoft SpaceV-100M][msft-spacev] subset | `msft-spacev-100m` |  `i8` |  100 |     L2 |   9 GB |  100 |
|                                             |                    |       |      |        |        |      |
| [Microsoft SpaceV-1B][msft-spacev]          | `msft-spacev-1b`   |  `i8` |  100 |     L2 | 131 GB |  100 |
| [Microsoft Turing-ANNS][msft-turing]        | `msft-turing-1b`   | `f32` |  100 |     L2 | 373 GB |  100 |
| [Yandex Deep1B][yandex-deep]                | `yandex-deep-1b`   | `f32` |   96 |     L2 | 358 GB |  100 |
| [Yandex Text-to-Image][yandex-t2i]          | `yandex-t2i-1b`    | `f32` |  200 |    Cos | 750 GB |  100 |
|                                             |                    |       |      |        |        |      |
| [ViT-L/12 LAION][laion]                     | `laion-5b`         | `f32` | 2048 |    Cos |  10 TB |    - |

Those often come with $N$ precomputed ground-truth neighbors, which is handy for recall evaluation.
The ground-truth neighbors are computed with respect to some "metric", such as Inner Product (IP), Cosine similarity (Cos), or Euclidean distance (L2).
They are generally distributed in a form of binary matrix files, packing either `f32` or `i8` scalars, for high compatibility, as opposed to less common `f16` and `bf16`.

[unum-cc-3m]: https://huggingface.co/datasets/unum-cloud/ann-cc-3m
[unum-wiki-1m]: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
[unum-t2i-1m]: https://huggingface.co/datasets/unum-cloud/ann-t2i-1m
[msft-spacev]: https://github.com/ashvardanian/SpaceV
[msft-turing]: https://learning2hash.github.io/publications/microsoftturinganns1B/
[yandex-t2i]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[yandex-deep]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[laion]: https://laion.ai/blog/laion-5b/#download-the-data

### Unum UForm Wiki

```sh
mkdir -p datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

### Yandex Text-to-Image

```sh
mkdir -p datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i_1B/
```

### Yandex Deep1B

```sh
mkdir -p datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_1B/
```

### Arxiv with E5

```sh
mkdir -p datasets/arxiv_2M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/abstract.e5-base-v2.fbin -P datasets/arxiv_2M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/title.e5-base-v2.fbin -P datasets/arxiv_2M/
```

### Microsoft SpaceV

The original dataset can be pulled in a USearch-compatible form from AWS S3:

```sh
mkdir -p datasets/spacev_1B/ && \
    aws s3 cp s3://bigger-ann/spacev-1b/ datasets/spacev_1B/ --recursive
```

A smaller 100M dataset can be pulled from Hugging Face:

```sh
mkdir -p datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/ids.100m.i32bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100m.i8bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.f32bin -P datasets/spacev_100M/
```

## Kubernetes Cluster Setup

For true distributed execution, create a local Kubernetes cluster using KIND (Kubernetes in Docker):

```bash
# Install KIND (if not already installed)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Create multi-node cluster
kind create cluster --config=k8s/kind-cluster.yaml --name=spark-benchmark

# Install Spark Operator
kubectl create namespace spark-operator
helm repo add spark-operator https://kubeflow.github.io/spark-operator
helm install spark-operator spark-operator/spark-operator --namespace spark-operator

# Build and load your application image
docker build -t usearch-spark:latest .
kind load docker-image usearch-spark:latest --name=spark-benchmark

# Submit Spark job
kubectl apply -f k8s/spark-job.yaml
kubectl logs -f spark-benchmark-driver
```

This creates a 3-node Kubernetes cluster locally and runs the benchmark in a truly distributed manner across the nodes.

## Contributing

```bash
gradle spotlessApply # in case you want to format changes before pushing
```
