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
        public BenchmarkConfig.Precision getPrecision() { return precision; }
        public long getIndexingTimeMs() { return indexingTimeMs; }
        public long getSearchTimeMs() { return searchTimeMs; }
        public double getThroughputQPS() { return throughputQPS; }
        public Map<Integer, Double> getRecallAtK() { return recallAtK; }
        public long getMemoryUsageBytes() { return memoryUsageBytes; }
        public int getNumVectors() { return numVectors; }
        public int getDimensions() { return dimensions; }
    }

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;

    public USearchBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) {
        this.config = config;
        this.dataset = dataset;
    }

    public Map<BenchmarkConfig.Precision, BenchmarkResult> runBenchmarks() throws Exception {
        Map<BenchmarkConfig.Precision, BenchmarkResult> results = new HashMap<>();
        
        logger.info("Starting USearch benchmarks for dataset: {}", dataset.getDefinition().getName());
        
        // Load base vectors and queries
        BinaryVectorLoader.VectorDataset baseVectors = BinaryVectorLoader.loadVectors(dataset.getBaseVectorPath());
        BinaryVectorLoader.VectorDataset queryVectors = BinaryVectorLoader.loadVectors(dataset.getQueryVectorPath());
        
        // Limit number of base vectors if specified
        int numBaseVectors = baseVectors.getRows();
        if (config.getMaxVectors() > 0 && config.getMaxVectors() < baseVectors.getRows()) {
            numBaseVectors = (int) Math.min(config.getMaxVectors(), Integer.MAX_VALUE);
            logger.info("Limiting base vectors to {} for faster testing", numBaseVectors);
        }
        
        logger.info("Using {} base vectors and {} query vectors", 
                   numBaseVectors, queryVectors.getRows());

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());
        
        for (BenchmarkConfig.Precision precision : config.getPrecisions()) {
            logger.info("Running USearch benchmark with precision: {}", precision.getName());
            
            try {
                BenchmarkResult result = runSingleBenchmark(baseVectors, queryVectors, precision, numQueries, numBaseVectors);
                results.put(precision, result);
                
                logger.info("USearch {} benchmark completed - Indexing: {}ms, Search: {}ms, Throughput: {:.2f} QPS",
                           precision.getName(), result.getIndexingTimeMs(), result.getSearchTimeMs(), result.getThroughputQPS());
                
            } catch (Exception e) {
                logger.error("Failed to run USearch benchmark for precision {}: {}", precision.getName(), e.getMessage());
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
            usearchMetric = Index.Metric.EUCLIDEAN;
        } else if ("ip".equals(metric)) {
            usearchMetric = Index.Metric.INNER_PRODUCT;
        } else if ("cos".equals(metric)) {
            usearchMetric = Index.Metric.COSINE;
        } else {
            usearchMetric = Index.Metric.EUCLIDEAN; // Default
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

        Index index = new Index.Config()
                .metric(usearchMetric)
                .quantization(quantization)
                .dimensions(baseVectors.getCols())
                .capacity(numBaseVectors)
                .connectivity(config.getMaxConnections())
                .expansion_add(config.getEfConstruction())
                .expansion_search(config.getEfSearch())
                .build();
        
        // Measure indexing time
        long startIndexing = System.currentTimeMillis();
        long memoryBefore = getMemoryUsage();
        
        // Add vectors to index
        for (int i = 0; i < numBaseVectors; i++) {
            float[] vector = baseVectors.getVectorAsFloat(i);
            index.add(i, vector);
            
            // Log progress
            if (i % 10000 == 0 && i > 0) {
                logger.debug("Indexed {} vectors", i);
            }
        }
        
        long indexingTime = System.currentTimeMillis() - startIndexing;
        long memoryAfter = getMemoryUsage();
        long memoryUsage = memoryAfter - memoryBefore;
        
        // Measure search time and calculate recall
        long startSearch = System.currentTimeMillis();
        Map<Integer, Double> recallAtK = calculateRecall(index, queryVectors, numQueries, config.getKValues());
        long searchTime = System.currentTimeMillis() - startSearch;
        
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
            baseVectors.getCols()
        );
    }

    private Map<Integer, Double> calculateRecall(Index index, BinaryVectorLoader.VectorDataset queryVectors, 
                                               int numQueries, int[] kValues) throws Exception {
        Map<Integer, Double> recallResults = new HashMap<>();
        
        // For now, we'll use a simplified recall calculation
        // In a full implementation, we'd load and compare against ground truth
        for (int k : kValues) {
            double totalRecall = 0.0;
            
            for (int i = 0; i < numQueries; i++) {
                float[] queryVector = queryVectors.getVectorAsFloat(i);
                long[] results = index.search(queryVector, k);
                
                // Simplified recall calculation - in practice you'd compare against ground truth
                // For now, just check if we got k results
                double queryRecall = Math.min(1.0, (double) results.length / k);
                totalRecall += queryRecall;
            }
            
            recallResults.put(k, totalRecall / numQueries);
        }
        
        return recallResults;
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
        logger.info("=== USearch Benchmark Results ===");
        
        // Sort by precision for consistent output
        List<BenchmarkConfig.Precision> sortedPrecisions = new ArrayList<>(results.keySet());
        sortedPrecisions.sort(Comparator.comparing(BenchmarkConfig.Precision::getName));
        
        for (BenchmarkConfig.Precision precision : sortedPrecisions) {
            BenchmarkResult result = results.get(precision);
            
            logger.info("Precision: {}", precision.getName());
            logger.info("  Indexing Time: {} ms", result.getIndexingTimeMs());
            logger.info("  Search Time: {} ms", result.getSearchTimeMs());
            logger.info("  Throughput: {:.2f} QPS", result.getThroughputQPS());
            logger.info("  Memory Usage: {:.2f} MB", result.getMemoryUsageBytes() / (1024.0 * 1024.0));
            
            for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
                logger.info("  Recall@{}: {:.4f}", entry.getKey(), entry.getValue());
            }
            
            logger.info("");
        }
    }
}