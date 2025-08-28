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
        private final String vectorIdsUrl; // Optional vector IDs file
        private final int dimensions;
        private final String metric;
        private final String format;
        private final long sizeBytes;

        public DatasetDefinition(String name, String description, String baseUrl, String queryUrl,
                String groundTruthUrl, int dimensions, String metric, String format, long sizeBytes) {
            this(name, description, baseUrl, queryUrl, groundTruthUrl, null, dimensions, metric, format, sizeBytes);
        }

        public DatasetDefinition(String name, String description, String baseUrl, String queryUrl,
                String groundTruthUrl, String vectorIdsUrl, int dimensions, String metric, String format, long sizeBytes) {
            this.name = name;
            this.description = description;
            this.baseUrl = baseUrl;
            this.queryUrl = queryUrl;
            this.groundTruthUrl = groundTruthUrl;
            this.vectorIdsUrl = vectorIdsUrl;
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

        public String getVectorIdsUrl() {
            return vectorIdsUrl;
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
        private final String vectorIdsPath;
        private final BinaryVectorLoader.DatasetInfo baseInfo;
        private final BinaryVectorLoader.DatasetInfo queryInfo;

        public Dataset(DatasetDefinition definition, String baseVectorPath, String queryVectorPath,
                String groundTruthPath, BinaryVectorLoader.DatasetInfo baseInfo,
                BinaryVectorLoader.DatasetInfo queryInfo) {
            this(definition, baseVectorPath, queryVectorPath, groundTruthPath, null, baseInfo, queryInfo);
        }

        public Dataset(DatasetDefinition definition, String baseVectorPath, String queryVectorPath,
                String groundTruthPath, String vectorIdsPath, BinaryVectorLoader.DatasetInfo baseInfo,
                BinaryVectorLoader.DatasetInfo queryInfo) {
            this.definition = definition;
            this.baseVectorPath = baseVectorPath;
            this.queryVectorPath = queryVectorPath;
            this.groundTruthPath = groundTruthPath;
            this.vectorIdsPath = vectorIdsPath;
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

        public String getVectorIdsPath() {
            return vectorIdsPath;
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
        // Small datasets (< 10GB) - good for testing
        DATASETS.put("unum-wiki-1m", new DatasetDefinition(
                "unum-wiki-1m",
                "UForm Wiki - 1GB, float32, 256 dimensions",
                "https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin",
                256, "ip", "fbin", 1_000_000_000L));

        DATASETS.put("unum-cc-3m", new DatasetDefinition(
                "unum-cc-3m",
                "UForm Creative Captions - 3GB, float32, 256 dimensions",
                "https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/base.3M.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/query.public.100K.fbin",
                "https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/groundtruth.public.100K.ibin",
                256, "ip", "fbin", 3_000_000_000L));

        DATASETS.put("yandex-t2i-1m", new DatasetDefinition(
                "yandex-t2i-1m",
                "Yandex Text-to-Image subset - 1GB, float32, 200 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin",
                200, "cos", "fbin", 1_000_000_000L));

        DATASETS.put("yandex-deep-10m", new DatasetDefinition(
                "yandex-deep-10m",
                "Yandex Deep10M - 4GB, float32, 96 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin",
                96, "l2", "fbin", 4_000_000_000L));

        DATASETS.put("msft-spacev-100m", new DatasetDefinition(
                "msft-spacev-100m",
                "Microsoft SpaceV subset - 9.3GB, int8, 100 dimensions",
                "https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100M.i8bin",
                "https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin",
                "https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin",
                "https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/ids.100M.i32bin",
                100, "l2", "i8bin", 9_300_000_000L));

        // Large-scale datasets (> 90GB) - billion-scale
        DATASETS.put("msft-spacev-1b", new DatasetDefinition(
                "msft-spacev-1b",
                "Microsoft SpaceV-1B - 131GB, int8, 100 dimensions",
                "s3://bigger-ann/spacev-1b/base.1B.i8bin",
                "s3://bigger-ann/spacev-1b/query.30K.i8bin",
                "s3://bigger-ann/spacev-1b/groundtruth.30K.i32bin",
                null, // No separate ID file for full 1B dataset
                100, "l2", "i8bin", 131_000_000_000L));

        DATASETS.put("msft-turing-1b", new DatasetDefinition(
                "msft-turing-1b",
                "Microsoft Turing-ANNS - 373GB, float32, 100 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/TURING/base.1B.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/TURING/query.100K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/TURING/groundtruth.100K.ibin",
                100, "l2", "fbin", 373_000_000_000L));

        DATASETS.put("yandex-deep-1b", new DatasetDefinition(
                "yandex-deep-1b",
                "Yandex Deep1B - 358GB, float32, 96 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin",
                96, "l2", "fbin", 358_000_000_000L));

        DATASETS.put("yandex-t2i-1b", new DatasetDefinition(
                "yandex-t2i-1b",
                "Yandex Text-to-Image - 750GB, float32, 200 dimensions",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
                "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin",
                200, "cos", "fbin", 750_000_000_000L));
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

        // Download vector IDs if available
        String vectorIdsPath = null;
        if (def.getVectorIdsUrl() != null && !def.getVectorIdsUrl().isEmpty()) {
            vectorIdsPath = downloadIfNeeded(def.getVectorIdsUrl(),
                    cachePath.resolve("ids" + getFileExtension(def.getVectorIdsUrl())).toString());
        }

        // Load dataset info
        BinaryVectorLoader.DatasetInfo baseInfo = null;
        BinaryVectorLoader.DatasetInfo queryInfo = null;

        try {
            if (baseVectorPath.endsWith(".fbin") || baseVectorPath.endsWith(".ibin") ||
                    baseVectorPath.endsWith(".dbin") || baseVectorPath.endsWith(".hbin") ||
                    baseVectorPath.endsWith(".bbin") || baseVectorPath.endsWith(".i8bin")) {
                baseInfo = BinaryVectorLoader.getDatasetInfo(baseVectorPath);
            }

            if (queryVectorPath.endsWith(".fbin") || queryVectorPath.endsWith(".ibin") ||
                    queryVectorPath.endsWith(".dbin") || queryVectorPath.endsWith(".hbin") ||
                    queryVectorPath.endsWith(".bbin") || queryVectorPath.endsWith(".i8bin")) {
                queryInfo = BinaryVectorLoader.getDatasetInfo(queryVectorPath);
            }
        } catch (IOException e) {
            logger.warn("Could not load dataset info: {}", e.getMessage());
        }

        return new Dataset(def, baseVectorPath, queryVectorPath, groundTruthPath, vectorIdsPath, baseInfo, queryInfo);
    }

    private static String downloadIfNeeded(String url, String localPath) throws IOException {
        Path path = Paths.get(localPath);

        if (Files.exists(path)) {
            logger.info("Using cached file: {}", localPath);
            return localPath;
        }

        // Handle S3 URLs specially
        if (url.startsWith("s3://")) {
            throw new IOException("S3 datasets require manual download.\n" +
                "Please run: aws s3 cp " + url + " " + localPath + "\n" +
                "Or use AWS CLI: aws configure && aws s3 sync s3://bigger-ann/spacev-1b/ datasets/");
        }

        logger.info("Downloading: {} -> {}", url, localPath);

        try (InputStream in = new URI(url).toURL().openStream()) {
            Files.copy(in, path, StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception e) {
            String errorMsg = "Failed to download from " + url + "\n" +
                "Possible causes:\n" +
                "  1. Network timeout - try again\n" +
                "  2. Large file - manually download: wget " + url + " -O " + localPath + "\n" +
                "  3. Access denied - check if dataset URL is still valid";
            throw new IOException(errorMsg, e);
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