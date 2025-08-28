package com.ashvardanian;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cloud.unum.usearch.Index;

/**
 * Multithreaded USearch benchmark with batch processing
 */
public class USearchBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(USearchBenchmark.class);

    public static class BenchmarkResult {
        private final BenchmarkConfig.Precision precision;
        private final long indexingTimeMs;
        private final long searchTimeMs;
        private final double throughputQPS;
        private final Map<Integer, Double> recallAtK;
        private final long memoryUsageBytes;
        private final int numVectors;
        private final int dimensions;

        public BenchmarkResult(BenchmarkConfig.Precision precision, long indexingTimeMs, long searchTimeMs,
                double throughputQPS, Map<Integer, Double> recallAtK, long memoryUsageBytes,
                int numVectors, int dimensions) {
            this.precision = precision;
            this.indexingTimeMs = indexingTimeMs;
            this.searchTimeMs = searchTimeMs;
            this.throughputQPS = throughputQPS;
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
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

        public Map<Integer, Double> getRecallAtK() {
            return recallAtK;
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

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;

    public USearchBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) {
        this.config = config;
        this.dataset = dataset;
    }

    public Map<BenchmarkConfig.Precision, BenchmarkResult> runBenchmarks() throws Exception {
        Map<BenchmarkConfig.Precision, BenchmarkResult> results = new HashMap<>();

        System.out.println("\n🔍 Starting USearch benchmarks for dataset: " + dataset.getDefinition().getName());

        // Load base vectors and queries
        System.out.print("📂 Loading vectors... ");
        BinaryVectorLoader.VectorDataset baseVectors = BinaryVectorLoader.loadVectors(dataset.getBaseVectorPath());
        BinaryVectorLoader.VectorDataset queryVectors = BinaryVectorLoader.loadVectors(dataset.getQueryVectorPath());
        System.out.println("✅ Done");

        // Limit number of base vectors if specified
        int numBaseVectors = baseVectors.getRows();
        if (config.getMaxVectors() > 0 && config.getMaxVectors() < baseVectors.getRows()) {
            numBaseVectors = (int) Math.min(config.getMaxVectors(), Integer.MAX_VALUE);
            System.out.println(
                    "🔢 Limiting base vectors to " + String.format("%,d", numBaseVectors) + " for faster testing");
        }

        System.out.println("📊 Using " + String.format("%,d", numBaseVectors) + " base vectors and " +
                String.format("%,d", queryVectors.getRows()) + " query vectors");

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());

        for (BenchmarkConfig.Precision precision : config.getPrecisions()) {
            System.out.println(String.format("\n⚙️ Running USearch benchmark with precision: %s", precision.getName()));

            try {
                BenchmarkResult result = runSingleBenchmark(baseVectors, queryVectors, precision, numQueries,
                        numBaseVectors);
                results.put(precision, result);

                System.out
                        .println(String.format("✅ %s completed - Indexing: %,dms, Search: %,dms, Throughput: %,.0f QPS",
                                precision.getName(), result.getIndexingTimeMs(), result.getSearchTimeMs(),
                                result.getThroughputQPS()));

            } catch (Exception e) {
                System.err.println(String.format("❌ Failed to run USearch benchmark for precision %s: %s",
                        precision.getName(), e.getMessage()));
                throw e;
            }
        }

        return results;
    }

    private BenchmarkResult runSingleBenchmark(BinaryVectorLoader.VectorDataset baseVectors,
            BinaryVectorLoader.VectorDataset queryVectors,
            BenchmarkConfig.Precision precision,
            int numQueries,
            int numBaseVectors) throws Exception {

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
            throw new IllegalArgumentException("Unsupported metric: " + metric + 
                ". Supported metrics are: l2, ip, cos");
        }

        // Set precision-specific quantization
        String quantization;
        boolean useByteData;
        switch (precision) {
            case F32:
                quantization = Index.Quantization.FLOAT32;
                useByteData = false;
                break;
            case F16:
                quantization = Index.Quantization.FLOAT16;
                useByteData = false;
                break;
            case BF16:
                quantization = Index.Quantization.BFLOAT16;
                useByteData = false;
                break;
            case I8:
                quantization = Index.Quantization.INT8;
                useByteData = true;
                break;
            default:
                quantization = Index.Quantization.FLOAT32;
                useByteData = false;
        }

        System.out.println(String.format("🔧 Creating USearch index: %,d vectors, %d dims, %s metric, %s precision",
                numBaseVectors, baseVectors.getCols(), usearchMetric, quantization));
        
        // Check available memory before creating index
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long freeMemory = runtime.freeMemory();
        long estimatedMemoryNeeded = (long) numBaseVectors * baseVectors.getCols() * (useByteData ? 1 : 4) * 2; // Rough estimate
        
        System.out.println(String.format("💾 Memory: Max: %,d MB, Free: %,d MB, Estimated needed: %,d MB",
                maxMemory / (1024 * 1024), freeMemory / (1024 * 1024), estimatedMemoryNeeded / (1024 * 1024)));
        
        if (estimatedMemoryNeeded > maxMemory * 0.8) {
            System.out.println("⚠️ Warning: Estimated memory usage is very high. Consider reducing --max-vectors");
        }

        try (Index index = createIndex(usearchMetric, quantization, baseVectors.getCols(), 
                numBaseVectors, config)) {

            // Measure indexing time
            long startIndexing = System.currentTimeMillis();
            long memoryBefore = getMemoryUsage();

            // Create batches for indexing
            List<VectorProcessor.VectorBatch> indexingBatches = VectorProcessor.createBatches(baseVectors,
                    numBaseVectors, useByteData, 1024);

            // batch indexing
            VectorProcessor.processBatches(indexingBatches, batch -> {
                if (batch.isByteData) {
                    // Add vectors individually with byte[] for I8
                    for (int i = 0; i < batch.vectorCount; i++) {
                        byte[] vector = new byte[batch.dimensions];
                        System.arraycopy(batch.byteVectors, i * batch.dimensions, vector, 0, batch.dimensions);
                        index.add(batch.keys[i], vector);
                    }
                } else {
                    // For F32/F16/BF16: Use batch add with concatenated vectors
                    // USearch auto-detects batch operations when array length > dimensions
                    index.add(batch.keys[0], batch.vectors); // Starts from first key, auto-increments
                }
                return null;
            }, "Indexing " + precision.getName());

            long indexingTime = System.currentTimeMillis() - startIndexing;
            long memoryAfter = getMemoryUsage();
            long memoryUsage = memoryAfter - memoryBefore;

            // Measure search time and calculate recall
            long startSearch = System.currentTimeMillis();
            System.out.print("🔍 Searching... ");
            Map<Integer, Double> recallAtK = calculateRecall(index, queryVectors, numQueries, config.getKValues(),
                    useByteData);
            long searchTime = System.currentTimeMillis() - startSearch;
            System.out.println("✅ Done");

            // Calculate throughput
            double throughputQPS = numQueries / (searchTime / 1000.0);

            return new BenchmarkResult(
                    precision,
                    indexingTime,
                    searchTime,
                    throughputQPS,
                    recallAtK,
                    memoryUsage,
                    numBaseVectors,
                    baseVectors.getCols());
        }
    }

    private Map<Integer, Double> calculateRecall(Index index, BinaryVectorLoader.VectorDataset queryVectors,
            int numQueries, int[] kValues, boolean useByteData) throws Exception {
        Map<Integer, Double> recallResults = new HashMap<>();

        // Try to load ground truth if available
        BinaryVectorLoader.GroundTruth groundTruth = null;
        try {
            String groundTruthPath = dataset.getGroundTruthPath();
            if (groundTruthPath != null && !groundTruthPath.isEmpty()) {
                groundTruth = BinaryVectorLoader.loadGroundTruth(groundTruthPath);
                logger.info("Using ground truth for accurate recall calculation");
            }
        } catch (Exception e) {
            logger.warn("Could not load ground truth, using simplified recall calculation: {}", e.getMessage());
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

        // Create batches for query vectors
        List<VectorProcessor.VectorBatch> queryBatches = 
            VectorProcessor.createBatches(queryVectors, numQueries, useByteData, 1024);

        final BinaryVectorLoader.GroundTruth finalGroundTruth = groundTruth;
        final BinaryVectorLoader.VectorIds finalVectorIds = vectorIds;

        // For each k value, run concurrent searches
        for (int k : kValues) {
            // Concurrent search processing
            List<Double> recalls = VectorProcessor.processBatches(queryBatches, batch -> {
                double batchRecall = 0.0;
                
                for (int i = 0; i < batch.vectorCount; i++) {
                    int globalQueryIndex = batch.startIndex + i;
                    
                    long[] results;
                    if (batch.isByteData) {
                        byte[] vector = new byte[batch.dimensions];
                        System.arraycopy(batch.byteVectors, i * batch.dimensions, vector, 0, batch.dimensions);
                        results = index.search(vector, k);
                    } else {
                        float[] vector = new float[batch.dimensions];
                        System.arraycopy(batch.vectors, i * batch.dimensions, vector, 0, batch.dimensions);
                        results = index.search(vector, k);
                    }
                    
                    // Calculate recall using ground truth if available
                    double queryRecall;
                    if (finalGroundTruth != null && globalQueryIndex < finalGroundTruth.getNumQueries()) {
                        // Convert long[] to int[] and map through vector IDs if available
                        int[] intResults = new int[results.length];
                        for (int j = 0; j < results.length; j++) {
                            int vectorIndex = (int) results[j];
                            // Map through vector IDs if available (for subset support)
                            if (finalVectorIds != null && vectorIndex < finalVectorIds.getNumVectors()) {
                                intResults[j] = finalVectorIds.getId(vectorIndex);
                            } else {
                                intResults[j] = vectorIndex;
                            }
                        }
                        queryRecall = BinaryVectorLoader.calculateRecallAtK(finalGroundTruth, globalQueryIndex, intResults, k);
                    } else {
                        // Simplified recall calculation
                        queryRecall = Math.min(1.0, (double) results.length / k);
                    }
                    
                    batchRecall += queryRecall;
                }
                
                return batchRecall;
            }, "Searching k=" + k);
            
            // Aggregate recall from all batches
            double totalRecall = recalls.stream().mapToDouble(Double::doubleValue).sum();
            recallResults.put(k, totalRecall / numQueries);
        }

        return recallResults;
    }

    private Index createIndex(String metric, String quantization, int dimensions, 
            int numVectors, BenchmarkConfig config) throws Exception {
        try {
            return new Index.Config()
                    .metric(metric)
                    .quantization(quantization)
                    .dimensions(dimensions)
                    .capacity(numVectors)
                    .connectivity(config.getMaxConnections())
                    .expansion_add(config.getEfConstruction())
                    .expansion_search(config.getEfSearch())
                    .build();
        } catch (Error e) {
            String errorMsg = String.format("Failed to create USearch index for %,d vectors.\n" +
                    "Suggestions:\n" +
                    "  1. Reduce vectors: gradle run --args=\"%s --max-vectors 100000\"\n" +
                    "  2. Increase heap: Add -Xmx16g to JAVA_OPTS\n" +
                    "  3. Use INT8 precision for lower memory usage\n" +
                    "Error: %s",
                    numVectors, System.getProperty("dataset.name", "dataset"), e.getMessage());
            throw new Exception(errorMsg, e);
        }
    }

    private long getMemoryUsage() {
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

        return runtime.totalMemory() - runtime.freeMemory();
    }

    public static void logBenchmarkResults(Map<BenchmarkConfig.Precision, BenchmarkResult> results) {
        System.out.println("=== USearch Benchmark Results ===");

        // Sort by precision for consistent output
        List<BenchmarkConfig.Precision> sortedPrecisions = new ArrayList<>(results.keySet());
        sortedPrecisions.sort(Comparator.comparing(BenchmarkConfig.Precision::getName));

        for (BenchmarkConfig.Precision precision : sortedPrecisions) {
            BenchmarkResult result = results.get(precision);

            System.out.println("Precision: " + precision.getName());
            System.out.println(String.format("  Indexing Time: %,d ms", result.getIndexingTimeMs()));
            System.out.println(String.format("  Search Time: %,d ms", result.getSearchTimeMs()));
            System.out.println(String.format("  Throughput: %,.0f QPS", result.getThroughputQPS()));
            System.out.println(String.format("  Memory Usage: %,d MB",
                    Math.round(result.getMemoryUsageBytes() / (1024.0 * 1024.0))));

            for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
                System.out.println(String.format("  Recall@%d: %.4f", entry.getKey(), entry.getValue()));
            }

            System.out.println();
        }
    }
}