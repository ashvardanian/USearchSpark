# USearchSpark

Vector Search benchmark comparing [USearch](https://github.com/unum-cloud/usearch) HNSW across precisions (`f32`, `f16`, `bf16`, `i8`) vs [Lucene](https://github.com/apache/lucene) HNSW (`f32`) baseline, leveraging Apache [Spark](https://github.com/apache/spark) for distributed indexing and search.

## Quick Start

```bash
gradle build --warning-mode all # ensure all dependencies are resolved
gradle run --args="unum-wiki-1m" # index & search `f32` vectors across all cores
gradle run --args="yandex-deep-10m --max-vectors 100000" # limit vectors for testing
gradle run --args="msft-spacev-100m --precision f32,i8" # test specific precisions
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

BigANN benchmark is a good starting point, if you are searching for large collections of high-dimensional vectors.
Those often come with precomputed ground-truth neighbors, which is handy for recall evaluation.

| Dataset                                     | Scalar Type | Dimensions | Metric |   Size    | Codename           |
| :------------------------------------------ | :---------: | :--------: | :----: | :-------: | :----------------- |
| [Unum UForm Creative Captions][unum-cc-3m]  |    `f32`    |    256     |   IP   |   3 GB    | `unum-cc-3m`       |
| [Unum UForm Wiki][unum-wiki-1m]             |    `f32`    |    256     |   IP   |   1 GB    | `unum-wiki-1m`     |
| [Yandex Text-to-Image][yandex-t2i] subset   |    `f32`    |    200     |  Cos   |   1 GB    | `yandex-t2i-1m`    |
| [Yandex Deep10M][yandex-deep] subset        |    `f32`    |     96     |   L2   |  358 GB   | `yandex-deep-10m`  |
| [Microsoft SpaceV-100M][msft-spacev] subset |    `i8`     |    100     |   L2   |  9.3 GB   | `msft-spacev-100m` |
|                                             |             |            |        |           |                    |
| [Microsoft SpaceV-1B][msft-spacev]          |    `i8`     |    100     |   L2   |   93 GB   | `msft-spacev-1b`   |
| [Microsoft Turing-ANNS][msft-turing]        |    `f32`    |    100     |   L2   |  373 GB   | `msft-turing-1b`   |
| [Yandex Deep1B][yandex-deep]                |    `f32`    |     96     |   L2   |  358 GB   | `yandex-deep-1b`   |
| [Yandex Text-to-Image][t2i]                 |    `f32`    |    200     |  Cos   |  750 GB   | `yandex-t2i-1b`    |
|                                             |             |            |        |           |                    |
| [ViT-L/12 LAION][laion]                     |    `f32`    |    2048    |  Cos   | 2 - 10 TB | `laion-5b`         |

Luckily, smaller samples of those datasets are available.

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
