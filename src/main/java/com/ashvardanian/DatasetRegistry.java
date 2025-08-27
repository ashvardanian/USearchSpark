package com.ashvardanian;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DatasetRegistry {
    private static final Logger logger = LoggerFactory.getLogger(DatasetRegistry.class);

    public static class DatasetDefinition {
        private final String name;
        private final String description;
        private final String baseUrl;
        private final String queryUrl;
        private final String groundTruthUrl;
        private final int dimensions;
        private final String metric;
        private final String format;
        private final long sizeBytes;

        public DatasetDefinition(String name, String description, String baseUrl, String queryUrl,
                String groundTruthUrl, int dimensions, String metric, String format, long sizeBytes) {
            this.name = name;
            this.description = description;
            this.baseUrl = baseUrl;
            this.queryUrl = queryUrl;
            this.groundTruthUrl = groundTruthUrl;
            this.dimensions = dimensions;
            this.metric = metric;
            this.format = format;
            this.sizeBytes = sizeBytes;
        }

        // Getters
        public String getName() {
            return name;
        }

        public String getDescription() {
            return description;
        }

        public String getBaseUrl() {
            return baseUrl;
        }

        public String getQueryUrl() {
            return queryUrl;
        }

        public String getGroundTruthUrl() {
            return groundTruthUrl;
        }

        public int getDimensions() {
            return dimensions;
        }

        public String getMetric() {
            return metric;
        }

        public String getFormat() {
            return format;
        }

        public long getSizeBytes() {
            return sizeBytes;
        }
    }

    public static class Dataset {
        private final DatasetDefinition definition;
        private final String baseVectorPath;
        private final String queryVectorPath;
        private final String groundTruthPath;
        private final BinaryVectorLoader.DatasetInfo baseInfo;
        private final BinaryVectorLoader.DatasetInfo queryInfo;

        public Dataset(DatasetDefinition definition, String baseVectorPath, String queryVectorPath,
                String groundTruthPath, BinaryVectorLoader.DatasetInfo baseInfo,
                BinaryVectorLoader.DatasetInfo queryInfo) {
            this.definition = definition;
            this.baseVectorPath = baseVectorPath;
            this.queryVectorPath = queryVectorPath;
            this.groundTruthPath = groundTruthPath;
            this.baseInfo = baseInfo;
            this.queryInfo = queryInfo;
        }

        public DatasetDefinition getDefinition() {
            return definition;
        }

        public String getBaseVectorPath() {
            return baseVectorPath;
        }

        public String getQueryVectorPath() {
            return queryVectorPath;
        }

        public String getGroundTruthPath() {
            return groundTruthPath;
        }

        public BinaryVectorLoader.DatasetInfo getBaseInfo() {
            return baseInfo;
        }

        public BinaryVectorLoader.DatasetInfo getQueryInfo() {
            return queryInfo;
        }
    }

    private static final Map<String, DatasetDefinition> DATASETS = new HashMap<>();
    private static final String DATASET_CACHE_DIR = "datasets";

    static {
        // UForm datasets from official USearch benchmarks (smaller, good for testing)
        DATASETS.put("wiki", new DatasetDefinition(
                "wiki",
                "UForm Wiki - 1GB, float32, 256 dimensions",
                "https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin",
                256, "ip", "fbin", 1_000_000_000L));

        DATASETS.put("creative-captions", new DatasetDefinition(
                "creative-captions",
                "UForm Creative Captions - 3GB, float32, 256 dimensions",
                "https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/base.3M.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/query.public.100K.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/groundtruth.public.100K.ibin",
                256, "ip", "fbin", 3_000_000_000L));

        DATASETS.put("text-to-image", new DatasetDefinition(
                "text-to-image",
                "Yandex Text-to-Image - 1GB, float32, 200 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin",
                200, "cos", "fbin", 1_000_000_000L));

        // Large-scale datasets from official USearch benchmarks (billion-scale)
        DATASETS.put("spacev-1b", new DatasetDefinition(
                "spacev-1b",
                "Microsoft SPACEV-1B - 131GB, int8, 100 dimensions (S3-backed)",
                "s3://spacev-1b/base.1B.i8bin",
                "s3://spacev-1b/query.30K.i8bin",
                "s3://spacev-1b/groundtruth.30K.i32bin",
                100, "l2", "i8bin", 131_000_000_000L));

        DATASETS.put("deep1b", new DatasetDefinition(
                "deep1b",
                "Yandex Deep1B - 358GB, float32, 96 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin",
                96, "l2", "fbin", 358_000_000_000L));

        // Smaller test dataset for development
        DATASETS.put("deep10m", new DatasetDefinition(
                "deep10m",
                "Yandex Deep10M - 4GB, float32, 96 dimensions (subset of Deep1B)",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin",
                96, "l2", "fbin", 4_000_000_000L));
    }

    public static Set<String> getAvailableDatasets() {
        return Collections.unmodifiableSet(DATASETS.keySet());
    }

    public static DatasetDefinition getDatasetDefinition(String name) {
        DatasetDefinition def = DATASETS.get(name.toLowerCase());
        if (def == null) {
            throw new IllegalArgumentException("Unknown dataset: " + name +
                    ". Available datasets: " + String.join(", ", getAvailableDatasets()));
        }
        return def;
    }

    public static Dataset loadDataset(String name) throws IOException {
        return loadDataset(name, DATASET_CACHE_DIR);
    }

    public static Dataset loadDataset(String name, String cacheDir) throws IOException {
        DatasetDefinition def = getDatasetDefinition(name);

        Path cachePath = Paths.get(cacheDir, name);
        Files.createDirectories(cachePath);

        logger.info("Loading dataset: {} ({})", def.getName(), def.getDescription());

        // Download files if they don't exist
        String baseVectorPath = downloadIfNeeded(def.getBaseUrl(),
                cachePath.resolve("base" + getFileExtension(def.getBaseUrl())).toString());
        String queryVectorPath = downloadIfNeeded(def.getQueryUrl(),
                cachePath.resolve("query" + getFileExtension(def.getQueryUrl())).toString());
        String groundTruthPath = downloadIfNeeded(def.getGroundTruthUrl(),
                cachePath.resolve("groundtruth" + getFileExtension(def.getGroundTruthUrl())).toString());

        // Load dataset info
        BinaryVectorLoader.DatasetInfo baseInfo = null;
        BinaryVectorLoader.DatasetInfo queryInfo = null;

        try {
            if (baseVectorPath.endsWith(".fbin") || baseVectorPath.endsWith(".ibin") ||
                    baseVectorPath.endsWith(".dbin") || baseVectorPath.endsWith(".hbin") ||
                    baseVectorPath.endsWith(".bbin")) {
                baseInfo = BinaryVectorLoader.getDatasetInfo(baseVectorPath);
            }

            if (queryVectorPath.endsWith(".fbin") || queryVectorPath.endsWith(".ibin") ||
                    queryVectorPath.endsWith(".dbin") || queryVectorPath.endsWith(".hbin") ||
                    queryVectorPath.endsWith(".bbin")) {
                queryInfo = BinaryVectorLoader.getDatasetInfo(queryVectorPath);
            }
        } catch (IOException e) {
            logger.warn("Could not load dataset info: {}", e.getMessage());
        }

        return new Dataset(def, baseVectorPath, queryVectorPath, groundTruthPath, baseInfo, queryInfo);
    }

    private static String downloadIfNeeded(String url, String localPath) throws IOException {
        Path path = Paths.get(localPath);

        if (Files.exists(path)) {
            logger.info("Using cached file: {}", localPath);
            return localPath;
        }

        logger.info("Downloading: {} -> {}", url, localPath);

        try (InputStream in = new URI(url).toURL().openStream()) {
            Files.copy(in, path, StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception e) {
            throw new IOException("Failed to download from " + url, e);
        }

        logger.info("Download completed: {}", localPath);
        return localPath;
    }

    private static String getFileExtension(String url) {
        int lastSlash = url.lastIndexOf('/');
        int lastDot = url.lastIndexOf('.');
        if (lastDot > lastSlash) {
            return url.substring(lastDot);
        }
        return "";
    }

    public static void printDatasetInfo() {
        logger.info("Available datasets:");
        for (DatasetDefinition def : DATASETS.values()) {
            logger.info("  {} - {} ({} dimensions, {} metric, {} GB)",
                    def.getName(),
                    def.getDescription(),
                    def.getDimensions(),
                    def.getMetric(),
                    def.getSizeBytes() / 1_000_000_000.0);
        }
    }

    public static boolean isValidDataset(String name) {
        return DATASETS.containsKey(name.toLowerCase());
    }
}