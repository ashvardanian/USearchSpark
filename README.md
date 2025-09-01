# USearchSpark

Vector Search benchmark comparing [USearch](https://github.com/unum-cloud/usearch) HNSW across precisions (`f32`, `f16`, `bf16`, `i8`) vs [Lucene](https://github.com/apache/lucene) in-memory HNSW (`f32`, `i8`) baselines, leveraging Apache [Spark](https://github.com/apache/spark) for distributed indexing and search.

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
│ Lucene       │ F32          │ 20,665       │ 864          │ 49.0 GB     │
│ Lucene       │ I8           │ 26,408       │ 1,218        │ 20.8 GB     │
│ USearch      │ F32          │ 96,119       │ 126,582      │ 96.0 GB     │
│ USearch      │ BF16         │ 113,090      │ 129,870      │ 64.0 GB     │
│ USearch      │ F16          │ 124,297      │ 144,928      │ 64.0 GB     │
│ USearch      │ I8           │ 137,329      │ 166,667      │ 48.0 GB     │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
🎯 RECALL & NDCG METRICS
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Engine      │ Precision   │ Recall@10   │ NDCG@10     │ Recall@100  │ NDCG@100    │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Lucene      │ F32         │ 90.00%      │ 88.17%      │ 94.70%      │ 93.34%      │
│ Lucene      │ I8          │ 90.00%      │ 87.75%      │ 94.97%      │ 93.30%      │
│ USearch     │ F32         │ 90.03%      │ 88.63%      │ 95.76%      │ 95.21%      │
│ USearch     │ BF16        │ 90.12%      │ 88.46%      │ 95.62%      │ 95.21%      │
│ USearch     │ F16         │ 90.27%      │ 88.59%      │ 95.78%      │ 95.31%      │
│ USearch     │ I8          │ 90.34%      │ 88.66%      │ 95.81%      │ 95.31%      │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

> __Hardware__: AWS `m7i.metal-48xl` instances (92 physical cores, 192 threads, 2 sockets).
> __OS__: Linux 6.8.0-1024-aws (Ubuntu 22.04.5 LTS).
> __CPU__: Intel Xeon Platinum 8488C @ 2.4GHz.
> __Memory__: 768 GB RAM total.
> __Java__: OpenJDK 21.0.5 with Java Vector API (`--add-modules=jdk.incubator.vector`).
> __JVM__: 128GB heap (`-Xmx128g`) with ZGC garbage collector (`-XX:+UseZGC`) for sub-10ms pauses.
> __Library versions__: USearch v2.20.8, Lucene v9.12.0.

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
mkdir -p datasets/unum-wiki-1m/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/unum-wiki-1m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/unum-wiki-1m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/unum-wiki-1m/
```

### Yandex Text-to-Image

```sh
mkdir -p datasets/yandex-t2i-1m/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/yandex-t2i-1m/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin -P datasets/yandex-t2i-1m/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/yandex-t2i-1m/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/yandex-t2i-1m/
```

### Yandex Deep1B

```sh
mkdir -p datasets/yandex-deep-1b/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/yandex-deep-1b/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin -P datasets/yandex-deep-1b/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/yandex-deep-1b/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/yandex-deep-1b/
```

### Microsoft SpaceV

The original dataset can be pulled in a USearch-compatible form from AWS S3:

```sh
mkdir -p datasets/msft-spacev-1b/ && \
    aws s3 cp s3://bigger-ann/spacev-1b/ datasets/msft-spacev-1b/ --recursive
```

A smaller 100M dataset can be pulled from Hugging Face:

```sh
mkdir -p datasets/msft-spacev-100m/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/ids.100m.i32bin -P datasets/msft-spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100m.i8bin -P datasets/msft-spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin -P datasets/msft-spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin -P datasets/msft-spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.f32bin -P datasets/msft-spacev-100m/
```

## TODO: Kubernetes Cluster Setup

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

- Code style: 120 chars/line using Eclipse JDT via Spotless. Config lives in `java-format.xml`.
- Formatting: run `gradle spotlessApply` before pushing changes.
- IDEs:
  - IntelliJ IDEA: import Eclipse formatter XML (Settings > Editor > Code Style > Java > Scheme > Import).
  - Eclipse: import `java-format.xml` (Preferences > Java > Code Style > Formatter > Import).

```bash
gradle spotlessApply
```

## Tuning Lucene

Many attempts have been made to tune Lucene and squeeze better numbers out of it.
In every case an improvement along one metric (IPS, QPS, Recall) has come at the cost of another.
That's common for search engines and suggests that we are approaching the limitations of the engine.

Here are some of the ideas considered and sometimes accepted:

- Using `StoredField` instead of `NumericDocValuesField` for IDs.
- Tuning the size and number of index segments to allow query-level parallelism.
- Fixed 4GB segment size instead of dynamic calculation based on CPU cores.
- Force merging segments when count exceeds optimal threshold.
- Eliminated thread explosion where 191 threads were used per individual query.
- Batch query processing using configurable `--batch-size` parameter.
- `TieredMergePolicy` resulted in extremely slow single-threaded reductions by the end.
- Replaced `ByteBuffersDirectory` with `DynamicByteBuffersDirectory` to handle >4GB indexes.
- Aggressive concurrent merge scheduling with progress tracking.
- Custom HNSW codec selection with fallback to best available implementation.
- Using `forceMerge` before search queries to compact & optimize the index.
- Committing the `IndexWriter` before search queries to trigger mergers.