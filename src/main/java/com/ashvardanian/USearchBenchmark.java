package com.ashvardanian;

import cloud.unum.usearch.Index;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Multithreaded USearch benchmark with batch processing */
public class USearchBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(USearchBenchmark.class);

    public static class BenchmarkResult {
        private final BenchmarkConfig.Precision precision;
        private final long indexingTimeMs;
        private final long searchTimeMs;
        private final double throughputQPS;
        private final Map<Integer, Double> recallAtK;
        private final Map<Integer, Double> ndcgAtK;
        private final long memoryUsageBytes;
        private final int numVectors;
        private final int dimensions;

        public BenchmarkResult(
                BenchmarkConfig.Precision precision,
                long indexingTimeMs,
                long searchTimeMs,
                double throughputQPS,
                Map<Integer, Double> recallAtK,
                Map<Integer, Double> ndcgAtK,
                long memoryUsageBytes,
                int numVectors,
                int dimensions) {
            this.precision = precision;
            this.indexingTimeMs = indexingTimeMs;
            this.searchTimeMs = searchTimeMs;
            this.throughputQPS = throughputQPS;
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.ndcgAtK = Collections.unmodifiableMap(new HashMap<>(ndcgAtK));
            this.memoryUsageBytes = memoryUsageBytes;
            this.numVectors = numVectors;
            this.dimensions = dimensions;
        }

        // Getters
        public BenchmarkConfig.Precision getPrecision() {
            return precision;
        }

        public long getIndexingTimeMs() {
            return indexingTimeMs;
        }

        public long getSearchTimeMs() {
            return searchTimeMs;
        }

        public double getThroughputQPS() {
            return throughputQPS;
        }

        public double getThroughputIPS() {
            return indexingTimeMs > 0 ? numVectors / (indexingTimeMs / 1000.0) : 0.0;
        }

        public Map<Integer, Double> getRecallAtK() {
            return recallAtK;
        }

        public Map<Integer, Double> getNDCGAtK() {
            return ndcgAtK;
        }

        public long getMemoryUsageBytes() {
            return memoryUsageBytes;
        }

        public int getNumVectors() {
            return numVectors;
        }

        public int getDimensions() {
            return dimensions;
        }
    }

    public static class SearchMetrics {
        public final Map<Integer, Double> recallAtK;
        public final Map<Integer, Double> ndcgAtK;
        public final long searchTimeMs;

        public SearchMetrics(
                Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK, long searchTimeMs) {
            this.recallAtK = recallAtK;
            this.ndcgAtK = ndcgAtK;
            this.searchTimeMs = searchTimeMs;
        }
    }

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;

    public USearchBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) {
        this.config = config;
        this.dataset = dataset;
    }

    public Map<BenchmarkConfig.Precision, BenchmarkResult> runBenchmarks() throws Exception {
        Map<BenchmarkConfig.Precision, BenchmarkResult> results = new HashMap<>();

        System.out.println(
                "\n🔍 Starting USearch benchmarks for dataset: "
                        + dataset.getDefinition().getName());

        // Load base vectors and queries with optional limits
        System.out.print("📂 Loading vectors... ");
        int maxBaseVectors =
                config.getMaxVectors() > 0
                        ? (int) Math.min(config.getMaxVectors(), Integer.MAX_VALUE)
                        : -1;
        BinaryVectorLoader.VectorDataset baseVectors =
                BinaryVectorLoader.loadVectors(dataset.getBaseVectorPath(), 0, maxBaseVectors);
        BinaryVectorLoader.VectorDataset queryVectors =
                BinaryVectorLoader.loadVectors(dataset.getQueryVectorPath());
        System.out.println("✅ Done");

        int numBaseVectors = baseVectors.getRows();
        if (maxBaseVectors > 0) {
            System.out.println(
                    "🔢 Limiting base vectors to "
                            + String.format("%,d", numBaseVectors)
                            + " for faster testing");
        }

        System.out.println(
                "📊 Using "
                        + String.format("%,d", numBaseVectors)
                        + " base vectors and "
                        + String.format("%,d", queryVectors.getRows())
                        + " query vectors");

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());
        for (BenchmarkConfig.Precision precision : config.getPrecisions()) {
            System.out.println(
                    String.format(
                            "\n⚙️ Running USearch benchmark with precision: %s",
                            precision.getName()));

            try {
                BenchmarkResult result =
                        runSingleBenchmark(
                                baseVectors, queryVectors, precision, numQueries, numBaseVectors);
                results.put(precision, result);

                System.out.println(
                        String.format(
                                "✅ %s completed - Indexing: %,dms, Search: %,dms, Throughput: %,.0f QPS",
                                precision.getName(),
                                result.getIndexingTimeMs(),
                                result.getSearchTimeMs(),
                                result.getThroughputQPS()));

            } catch (Exception e) {
                System.err.println(
                        String.format(
                                "❌ Failed to run USearch benchmark for precision %s: %s",
                                precision.getName(), e.getMessage()));
                throw e;
            }
        }

        return results;
    }

    private BenchmarkResult runSingleBenchmark(
            BinaryVectorLoader.VectorDataset baseVectors,
            BinaryVectorLoader.VectorDataset queryVectors,
            BenchmarkConfig.Precision precision,
            int numQueries,
            int numBaseVectors)
            throws Exception {

        // Set metric based on dataset
        String metric = dataset.getDefinition().getMetric().toLowerCase();
        String usearchMetric;
        if ("l2".equals(metric)) {
            usearchMetric = Index.Metric.EUCLIDEAN_SQUARED;
        } else if ("ip".equals(metric)) {
            usearchMetric = Index.Metric.INNER_PRODUCT;
        } else if ("cos".equals(metric)) {
            usearchMetric = Index.Metric.COSINE;
        } else {
            throw new IllegalArgumentException(
                    "Unsupported metric: " + metric + ". Supported metrics are: l2, ip, cos");
        }

        // Set precision-specific quantization
        String quantization;
        switch (precision) {
            case F32:
                quantization = Index.Quantization.FLOAT32;
                break;
            case F16:
                quantization = Index.Quantization.FLOAT16;
                break;
            case BF16:
                quantization = Index.Quantization.BFLOAT16;
                break;
            case I8:
                quantization = Index.Quantization.INT8;
                break;
            default:
                quantization = Index.Quantization.FLOAT32;
        }

        // Determine input format based on dataset, NOT quantization
        BinaryVectorLoader.VectorType vectorType = baseVectors.getType();
        boolean useByteData =
                vectorType == BinaryVectorLoader.VectorType.INT8
                        || vectorType == BinaryVectorLoader.VectorType.UINT8
                        || vectorType == BinaryVectorLoader.VectorType.UINT8_BIN;

        System.out.println(
                String.format(
                        "🔧 Creating USearch index: %,d vectors, %d dims, %s metric, %s precision",
                        numBaseVectors, baseVectors.getCols(), usearchMetric, quantization));

        // Check available memory before creating index
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long freeMemory = runtime.freeMemory();
        long estimatedMemoryNeeded =
                (long) numBaseVectors * baseVectors.getCols() * (useByteData ? 1 : 4) * 2; // Rough
        // estimate

        System.out.println(
                String.format(
                        "💾 Memory: Max: %,d MB, Free: %,d MB, Estimated needed: %,d MB",
                        maxMemory / (1024 * 1024),
                        freeMemory / (1024 * 1024),
                        estimatedMemoryNeeded / (1024 * 1024)));

        if (estimatedMemoryNeeded > maxMemory * 0.8) {
            System.out.println(
                    "⚠️ Warning: Estimated memory usage is very high. Consider reducing --max-vectors");
        }

        String version = Index.version();
        boolean usesDynamicDispatch = Index.usesDynamicDispatch();
        String[] compiled = Index.hardwareAccelerationCompiled();
        String[] available = Index.hardwareAccelerationAvailable();

        System.out.println("📦 Library version: " + version);
        System.out.println("🎯 Uses dynamic dispatch: " + usesDynamicDispatch);
        System.out.println("⚙️ Compiled capabilities: " + String.join(", ", compiled));
        System.out.println("🔧 Available capabilities: " + String.join(", ", available));

        try (Index index =
                createIndex(
                        usearchMetric,
                        quantization,
                        baseVectors.getCols(),
                        numBaseVectors,
                        config)) {

            // Log hardware acceleration info
            System.out.println("🚀 Hardware acceleration: " + index.hardwareAcceleration());

            // Measure indexing time
            long startIndexing = System.currentTimeMillis();
            long memoryBefore = getMemoryUsage(index);

            // Determine optimal thread count (cap at reasonable number for JNI overhead)
            int numThreads =
                    config.getNumThreads() != -1
                            ? config.getNumThreads()
                            : ForkJoinPool.commonPool().getParallelism();
            System.out.println(
                    "🧵 Using "
                            + numThreads
                            + " threads for indexing ("
                            + ForkJoinPool.commonPool().getParallelism()
                            + " available)");

            // Parallel indexing using ForkJoinPool
            ProgressLogger indexProgress =
                    new ProgressLogger("Indexing " + precision.getName(), numBaseVectors);

            if (numThreads == 1) {
                // Single-threaded fallback - reuse buffer
                float[] floatBuffer = useByteData ? null : new float[baseVectors.getCols()];
                byte[] byteBuffer = useByteData ? new byte[baseVectors.getCols()] : null;

                for (int i = 0; i < numBaseVectors; i++) {
                    if (useByteData) {
                        baseVectors.getVectorAsByte(i, byteBuffer);
                        index.add(i, byteBuffer);
                    } else {
                        baseVectors.getVectorAsFloat(i, floatBuffer);
                        index.add(i, floatBuffer);
                    }
                    indexProgress.increment();
                }
            } else {
                // Multi-threaded indexing with clean work partitioning
                ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
                try {
                    List<CompletableFuture<Void>> futures = new ArrayList<>();

                    // Partition work evenly across threads
                    int vectorsPerThread = numBaseVectors / numThreads;
                    int remainingVectors = numBaseVectors % numThreads;

                    for (int threadId = 0; threadId < numThreads; threadId++) {
                        final int startIdx =
                                threadId * vectorsPerThread + Math.min(threadId, remainingVectors);
                        final int endIdx =
                                startIdx + vectorsPerThread + (threadId < remainingVectors ? 1 : 0);
                        final int finalThreadId = threadId;

                        CompletableFuture<Void> future =
                                CompletableFuture.runAsync(
                                        () -> {
                                            try {
                                                if (useByteData) {
                                                    byte[] byteBuffer =
                                                            new byte[baseVectors.getCols()];
                                                    for (int i = startIdx; i < endIdx; i++) {
                                                        baseVectors.getVectorAsByte(i, byteBuffer);
                                                        index.add(i, byteBuffer);
                                                        indexProgress.increment();
                                                    }
                                                } else {
                                                    float[] floatBuffer =
                                                            new float[baseVectors.getCols()];
                                                    for (int i = startIdx; i < endIdx; i++) {
                                                        baseVectors.getVectorAsFloat(
                                                                i, floatBuffer);
                                                        index.add(i, floatBuffer);
                                                        indexProgress.increment();
                                                    }
                                                }
                                            } catch (Exception e) {
                                                throw new RuntimeException(
                                                        "Indexing failed in thread "
                                                                + finalThreadId
                                                                + " (range "
                                                                + startIdx
                                                                + "-"
                                                                + endIdx
                                                                + ")",
                                                        e);
                                            }
                                        },
                                        customThreadPool);

                        futures.add(future);
                    }

                    // Wait for all threads to complete
                    CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

                } finally {
                    customThreadPool.shutdown();
                }
            }

            indexProgress.complete(numBaseVectors);

            long indexingTime = System.currentTimeMillis() - startIndexing;
            long memoryAfter = getMemoryUsage(index);
            // Use Math.max to prevent negative memory usage due to GC
            long memoryUsage = Math.max(0, memoryAfter - memoryBefore);

            // Calculate search metrics (includes both search and accuracy calculation)
            System.out.print("🔍 Searching... ");
            SearchMetrics metrics =
                    calculateSearchMetrics(
                            index, queryVectors, numQueries, config.getKValues(), useByteData);
            System.out.println("✅ Done");

            // Use the actual search time from metrics (excludes accuracy calculation)
            long searchTime = metrics.searchTimeMs;

            // Calculate throughput based on actual search time
            double throughputQPS = numQueries / (searchTime / 1000.0);

            return new BenchmarkResult(
                    precision,
                    indexingTime,
                    searchTime,
                    throughputQPS,
                    metrics.recallAtK,
                    metrics.ndcgAtK,
                    memoryUsage,
                    numBaseVectors,
                    baseVectors.getCols());
        }
    }

    private SearchMetrics calculateSearchMetrics(
            Index index,
            BinaryVectorLoader.VectorDataset queryVectors,
            int numQueries,
            int[] kValues,
            boolean useByteData)
            throws Exception {

        // Try to load ground truth if available
        BinaryVectorLoader.GroundTruth groundTruth = null;
        try {
            String groundTruthPath = dataset.getGroundTruthPath();
            if (groundTruthPath != null && !groundTruthPath.isEmpty()) {
                groundTruth = BinaryVectorLoader.loadGroundTruth(groundTruthPath);
                logger.info("Using ground truth for accurate recall calculation");
            }
        } catch (Exception e) {
            logger.warn(
                    "Could not load ground truth, using simplified recall calculation: {}",
                    e.getMessage());
        }

        if (groundTruth == null) {
            System.out.println("⚠️ No ground truth available - accuracy metrics will be zero");
        }

        // Try to load vector IDs if available
        BinaryVectorLoader.VectorIds vectorIds = null;
        try {
            String vectorIdsPath = dataset.getVectorIdsPath();
            if (vectorIdsPath != null && !vectorIdsPath.isEmpty()) {
                vectorIds = BinaryVectorLoader.loadVectorIds(vectorIdsPath);
                logger.info("Using vector ID mapping for subset support");
            }
        } catch (Exception e) {
            logger.warn("Could not load vector IDs: {}", e.getMessage());
        }

        // Pre-allocate result arrays to avoid allocation overhead during timing
        long[][] allSearchResults = new long[numQueries][];

        // Find maximum K value for single search optimization
        int maxK = Arrays.stream(kValues).max().orElse(100);

        // Use all available threads for search
        int numThreads =
                config.getNumThreads() != -1
                        ? config.getNumThreads()
                        : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();
        System.out.println("🔍 Using " + numThreads + " threads for search");

        // SINGLE SEARCH with maximum K - no accuracy calculations during timing
        ProgressLogger searchProgress = new ProgressLogger("Searching k=" + maxK, numQueries);
        long startSearch = System.currentTimeMillis();

        if (numThreads == 1) {
            // Single-threaded fallback - reuse buffers
            float[] queryFloatBuffer = useByteData ? null : new float[queryVectors.getCols()];
            byte[] queryByteBuffer = useByteData ? new byte[queryVectors.getCols()] : null;

            for (int i = 0; i < numQueries; i++) {
                if (useByteData) {
                    queryVectors.getVectorAsByte(i, queryByteBuffer);
                    allSearchResults[i] = index.search(queryByteBuffer, maxK);
                } else {
                    queryVectors.getVectorAsFloat(i, queryFloatBuffer);
                    allSearchResults[i] = index.search(queryFloatBuffer, maxK);
                }
                searchProgress.increment();
            }
        } else {
            // Multi-threaded search with clean work partitioning
            ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
            try {
                List<CompletableFuture<Void>> futures = new ArrayList<>();

                // Partition work evenly across threads
                int queriesPerThread = numQueries / numThreads;
                int remainingQueries = numQueries % numThreads;

                for (int threadId = 0; threadId < numThreads; threadId++) {
                    final int startIdx =
                            threadId * queriesPerThread + Math.min(threadId, remainingQueries);
                    final int endIdx =
                            startIdx + queriesPerThread + (threadId < remainingQueries ? 1 : 0);
                    final int finalThreadId = threadId;

                    CompletableFuture<Void> future =
                            CompletableFuture.runAsync(
                                    () -> {
                                        try {
                                            if (useByteData) {
                                                byte[] queryBuffer =
                                                        new byte[queryVectors.getCols()];
                                                for (int i = startIdx; i < endIdx; i++) {
                                                    queryVectors.getVectorAsByte(i, queryBuffer);
                                                    allSearchResults[i] =
                                                            index.search(queryBuffer, maxK);
                                                    searchProgress.increment();
                                                }
                                            } else {
                                                float[] queryBuffer =
                                                        new float[queryVectors.getCols()];
                                                for (int i = startIdx; i < endIdx; i++) {
                                                    queryVectors.getVectorAsFloat(i, queryBuffer);
                                                    allSearchResults[i] =
                                                            index.search(queryBuffer, maxK);
                                                    searchProgress.increment();
                                                }
                                            }
                                        } catch (Exception e) {
                                            throw new RuntimeException(
                                                    "Search failed in thread "
                                                            + finalThreadId
                                                            + " (range "
                                                            + startIdx
                                                            + "-"
                                                            + endIdx
                                                            + ")",
                                                    e);
                                        }
                                    },
                                    customThreadPool);

                    futures.add(future);
                }

                // Wait for all threads to complete
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

            } finally {
                customThreadPool.shutdown();
            }
        }

        long searchTime = System.currentTimeMillis() - startSearch;
        searchProgress.complete(numQueries);

        // ACCURACY CALCULATION for all K values from single search results (after
        // timing)
        System.out.println("📊 Calculating accuracy metrics...");

        // Initialize result maps for all K values
        Map<Integer, Double> recallResults = new HashMap<>();
        Map<Integer, Double> ndcgResults = new HashMap<>();

        for (int k : kValues) {
            double totalRecall = 0.0;
            double totalNdcg = 0.0;
            int validQueries =
                    groundTruth != null ? Math.min(numQueries, groundTruth.getNumQueries()) : 0;

            for (int i = 0; i < validQueries; i++) {
                long[] allResults = allSearchResults[i];
                int actualK = Math.min(k, allResults.length);
                long[] resultsAtK = Arrays.copyOf(allResults, actualK);

                // Convert long[] to int[] and map through vector IDs if available
                int[] intResults = new int[actualK];
                for (int j = 0; j < actualK; j++) {
                    int vectorIndex = (int) resultsAtK[j];
                    // Map through vector IDs if available (for subset support)
                    if (vectorIds != null && vectorIndex < vectorIds.getNumVectors()) {
                        intResults[j] = vectorIds.getId(vectorIndex);
                    } else {
                        intResults[j] = vectorIndex;
                    }
                }

                totalRecall += BinaryVectorLoader.calculateRecallAtK(groundTruth, i, intResults, k);
                totalNdcg += BinaryVectorLoader.calculateNDCGAtK(groundTruth, i, intResults, k);
            }

            recallResults.put(k, validQueries > 0 ? totalRecall / validQueries : 0.0);
            ndcgResults.put(k, validQueries > 0 ? totalNdcg / validQueries : 0.0);
        }

        return new SearchMetrics(recallResults, ndcgResults, searchTime);
    }

    private Index createIndex(
            String metric,
            String quantization,
            int dimensions,
            int numVectors,
            BenchmarkConfig config)
            throws Exception {
        try {
            return new Index.Config()
                    .metric(metric)
                    .quantization(quantization)
                    .dimensions(dimensions)
                    .capacity(numVectors)
                    .build();
        } catch (Error e) {
            String errorMsg =
                    String.format(
                            "Failed to create USearch index for %,d vectors.\n"
                                    + "Suggestions:\n"
                                    + "  1. Reduce vectors: gradle run --args=\"%s --max-vectors 100000\"\n"
                                    + "  2. Increase heap: Add -Xmx16g to JAVA_OPTS\n"
                                    + "  3. Use INT8 precision for lower memory usage\n"
                                    + "Error: %s",
                            numVectors,
                            System.getProperty("dataset.name", "dataset"),
                            e.getMessage());
            throw new Exception(errorMsg, e);
        }
    }

    private long getMemoryUsage(Index index) {
        if (!config.isIncludeMemoryUsage()) {
            return 0;
        }

        Runtime runtime = Runtime.getRuntime();
        runtime.gc(); // Suggest garbage collection

        try {
            Thread.sleep(100); // Give GC a moment
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        long jvmMemory = runtime.totalMemory() - runtime.freeMemory();
        long nativeMemory = index.memoryUsage();

        return jvmMemory + nativeMemory;
    }

    public static void logBenchmarkResults(
            Map<BenchmarkConfig.Precision, BenchmarkResult> results) {
        System.out.println("=== USearch Benchmark Results ===");

        // Sort by precision for consistent output
        List<BenchmarkConfig.Precision> sortedPrecisions = new ArrayList<>(results.keySet());
        sortedPrecisions.sort(Comparator.comparing(BenchmarkConfig.Precision::getName));

        for (BenchmarkConfig.Precision precision : sortedPrecisions) {
            BenchmarkResult result = results.get(precision);

            System.out.println("Precision: " + precision.getName());
            System.out.println(
                    String.format("  Indexing Time: %,d ms", result.getIndexingTimeMs()));
            System.out.println(String.format("  Search Time: %,d ms", result.getSearchTimeMs()));
            System.out.println(
                    String.format("  Indexing Throughput: %,.0f IPS", result.getThroughputIPS()));
            System.out.println(
                    String.format("  Search Throughput: %,.0f QPS", result.getThroughputQPS()));
            System.out.println(
                    String.format(
                            "  Memory Usage: %,d MB",
                            Math.round(result.getMemoryUsageBytes() / (1024.0 * 1024.0))));

            for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
                int k = entry.getKey();
                double recall = entry.getValue() * 100.0; // Convert to percentage
                double ndcg =
                        result.getNDCGAtK().getOrDefault(k, 0.0) * 100.0; // Convert to percentage
                System.out.println(
                        String.format("  Recall@%d: %.2f%%, NDCG@%d: %.2f%%", k, recall, k, ndcg));
            }

            System.out.println();
        }
    }
}
