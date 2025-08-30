package com.ashvardanian;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SparkBenchmarkCoordinator {
    private static final Logger logger = LoggerFactory.getLogger(SparkBenchmarkCoordinator.class);

    public static class BenchmarkResults {
        private final String datasetName;
        private final BenchmarkConfig.BenchmarkMode mode;
        private final long totalTimeMs;
        private final Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults;
        private final LuceneBenchmark.BenchmarkResult luceneResult;
        private final long timestamp;

        public BenchmarkResults(String datasetName, BenchmarkConfig.BenchmarkMode mode, long totalTimeMs,
                Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults,
                LuceneBenchmark.BenchmarkResult luceneResult) {
            this.datasetName = datasetName;
            this.mode = mode;
            this.totalTimeMs = totalTimeMs;
            this.usearchResults = Collections.unmodifiableMap(new HashMap<>(usearchResults));
            this.luceneResult = luceneResult;
            this.timestamp = System.currentTimeMillis();
        }

        // Getters
        public String getDatasetName() {
            return datasetName;
        }

        public BenchmarkConfig.BenchmarkMode getMode() {
            return mode;
        }

        public long getTotalTimeMs() {
            return totalTimeMs;
        }

        public Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> getUsearchResults() {
            return usearchResults;
        }

        public LuceneBenchmark.BenchmarkResult getLuceneResult() {
            return luceneResult;
        }

        public long getTimestamp() {
            return timestamp;
        }

    }

    private final SparkSession spark;
    private ScheduledExecutorService progressScheduler;
    private LongAccumulator progressAccumulator;

    public SparkBenchmarkCoordinator(SparkSession spark) {
        this.spark = spark;
        this.progressAccumulator = spark.sparkContext().longAccumulator("Benchmark Progress");
    }

    public BenchmarkResults runBenchmark(BenchmarkConfig config) throws Exception {
        logger.info("Starting distributed vector search benchmark");
        logger.info("Dataset: {}", config.getDatasetName());
        logger.info("Mode: {}", config.getMode());
        logger.info("Output: {}", config.getOutputPath());

        long startTime = System.currentTimeMillis();

        try {
            // Start progress tracking
            startProgressTracking();

            // Load dataset
            DatasetRegistry.Dataset dataset = loadDataset(config);

            // Run benchmark based on mode (local or distributed)
            BenchmarkResults results = runBenchmark(config, dataset);

            // Save results
            saveResults(config, results);

            // Display comparison table
            printBenchmarkComparison(results.usearchResults, results.luceneResult);

            long totalTime = System.currentTimeMillis() - startTime;
            logger.info("Benchmark completed in {} ms", totalTime);

            return results;

        } finally {
            stopProgressTracking();
        }
    }

    private DatasetRegistry.Dataset loadDataset(BenchmarkConfig config) throws IOException {
        logger.info("Loading dataset: {}", config.getDatasetName());

        if (!DatasetRegistry.isValidDataset(config.getDatasetName())) {
            logger.error("Unknown dataset: {}", config.getDatasetName());
            DatasetRegistry.printDatasetInfo();
            throw new IllegalArgumentException("Unknown dataset: " + config.getDatasetName());
        }

        progressAccumulator.add(10); // Dataset loading progress
        return DatasetRegistry.loadDataset(config.getDatasetName());
    }

    private BenchmarkResults runBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) throws Exception {

        if (config.getMode() == BenchmarkConfig.BenchmarkMode.LOCAL) {
            return runLocalBenchmark(config, dataset);
        } else {
            return runDistributedBenchmark(config, dataset);
        }
    }

    private BenchmarkResults runLocalBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset)
            throws Exception {
        logger.info("Running local (single-node) benchmark");

        // Run Lucene benchmark
        logger.info("Running Lucene benchmark");
        progressAccumulator.add(20);
        LuceneBenchmark luceneBenchmark = new LuceneBenchmark(config, dataset);
        LuceneBenchmark.BenchmarkResult luceneResult = luceneBenchmark.runBenchmark();

        // Run USearch benchmarks for each precision
        progressAccumulator.add(30);
        Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults = new HashMap<>();

        for (BenchmarkConfig.Precision precision : config.getPrecisions()) {
            logger.info("Running USearch {} benchmark", precision.getName());
            // Run single precision benchmark
            USearchBenchmark usearchBenchmark = new USearchBenchmark(config, dataset);
            USearchBenchmark.BenchmarkResult result = usearchBenchmark.runBenchmark(precision);
            usearchResults.put(precision, result);
        }

        progressAccumulator.add(40);

        // Log results with clean output
        USearchBenchmark.logBenchmarkResults(usearchResults);
        LuceneBenchmark.logBenchmarkResults(luceneResult);

        long totalTime = System.currentTimeMillis();

        return new BenchmarkResults(config.getDatasetName(), config.getMode(), totalTime, usearchResults, luceneResult);
    }

    private BenchmarkResults runDistributedBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset)
            throws Exception {
        logger.info("Running distributed (Spark cluster) benchmark");

        // Run distributed Lucene benchmark first
        progressAccumulator.add(20);
        LuceneSparkBenchmark luceneSparkBenchmark = new LuceneSparkBenchmark(config, dataset, spark);
        LuceneSparkBenchmark.BenchmarkResult luceneSparkResult = luceneSparkBenchmark.runBenchmark();

        // Log Lucene results immediately
        LuceneSparkBenchmark.logBenchmarkResults(luceneSparkResult);

        // Force cleanup after Lucene benchmark
        System.out.println("🧹 Cleaning up memory after Lucene benchmark...");
        luceneSparkBenchmark = null; // Release reference
        System.gc(); // Suggest garbage collection
        try {
            Thread.sleep(2000); // Give GC time to work
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Log memory status
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory() / (1024 * 1024);
        long freeMemory = runtime.freeMemory() / (1024 * 1024);
        long maxMemory = runtime.maxMemory() / (1024 * 1024);
        System.out.println(String.format("💾 Memory after cleanup: %,d MB used, %,d MB free, %,d MB max",
                totalMemory - freeMemory, freeMemory, maxMemory));

        // Run distributed USearch benchmarks
        progressAccumulator.add(30);
        USearchSparkBenchmark usearchSparkBenchmark = new USearchSparkBenchmark(config, dataset, spark);
        Map<BenchmarkConfig.Precision, USearchSparkBenchmark.BenchmarkResult> usearchSparkResults = usearchSparkBenchmark
                .runBenchmarks();
        progressAccumulator.add(40);

        // Log USearch results
        USearchSparkBenchmark.logBenchmarkResults(usearchSparkResults);

        long totalTime = System.currentTimeMillis();

        // Convert distributed results to single-node format for unified interface
        Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> convertedUSearchResults = convertUSearchSparkResults(
                usearchSparkResults);
        LuceneBenchmark.BenchmarkResult convertedLuceneResult = convertLuceneSparkResult(luceneSparkResult);

        return new BenchmarkResults(config.getDatasetName(), config.getMode(), totalTime, convertedUSearchResults,
                convertedLuceneResult);
    }

    private void printBenchmarkComparison(
            Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults,
            LuceneBenchmark.BenchmarkResult luceneResult) {
        // Use the new AsciiTable-based comparison table
        BenchmarkResultsTable.printComparisonTable(usearchResults, luceneResult);
    }

    private void saveResults(BenchmarkConfig config, BenchmarkResults results) throws IOException {
        // Create output directory
        Path outputPath = Paths.get(config.getOutputPath());
        Files.createDirectories(outputPath);

        // Save as JSON
        ObjectMapper objectMapper = new ObjectMapper();
        ObjectWriter writer = objectMapper.writerWithDefaultPrettyPrinter();

        String filename = String.format("stats_%s_%s_%d.json", results.getDatasetName(), results.getMode().getName(),
                results.getTimestamp());

        Path resultsFile = outputPath.resolve(filename);
        writer.writeValue(resultsFile.toFile(), results);

        logger.info("Results saved to: {}", resultsFile.toAbsolutePath());

        // Also save a summary CSV for easy analysis
        saveSummaryCSV(outputPath, results);
    }

    private void saveSummaryCSV(Path outputPath, BenchmarkResults results) throws IOException {
        Path csvFile = outputPath.resolve("benchmark_summary.csv");
        boolean fileExists = Files.exists(csvFile);

        try (FileWriter fw = new FileWriter(csvFile.toFile(), true); PrintWriter pw = new PrintWriter(fw)) {

            // Write header if file doesn't exist
            if (!fileExists) {
                pw.println(
                        "timestamp,dataset,mode,implementation,precision,indexing_time_ms,search_time_ms,throughput_qps,memory_usage_mb,recall_at_1,recall_at_10,recall_at_100");
            }

            // Write USearch results
            for (Map.Entry<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> entry : results
                    .getUsearchResults().entrySet()) {

                USearchBenchmark.BenchmarkResult result = entry.getValue();
                pw.printf("%d,%s,%s,USearch,%s,%d,%d,%.2f,%.2f,%.4f,%.4f,%.4f%n", results.getTimestamp(),
                        results.getDatasetName(), results.getMode().getName(), entry.getKey().getName(),
                        result.getIndexingTimeMs(), result.getSearchTimeMs(), result.getThroughputQPS(),
                        result.getMemoryUsageBytes() / (1024.0 * 1024.0), result.getRecallAtK().getOrDefault(1, 0.0),
                        result.getRecallAtK().getOrDefault(10, 0.0), result.getRecallAtK().getOrDefault(100, 0.0));
            }

            // Write Lucene result
            LuceneBenchmark.BenchmarkResult luceneResult = results.getLuceneResult();
            pw.printf("%d,%s,%s,Lucene,F32,%d,%d,%.2f,%.2f,%.4f,%.4f,%.4f%n", results.getTimestamp(),
                    results.getDatasetName(), results.getMode().getName(), luceneResult.getIndexingTimeMs(),
                    luceneResult.getSearchTimeMs(), luceneResult.getThroughputQPS(),
                    luceneResult.getMemoryUsageBytes() / (1024.0 * 1024.0),
                    luceneResult.getRecallAtK().getOrDefault(1, 0.0), luceneResult.getRecallAtK().getOrDefault(10, 0.0),
                    luceneResult.getRecallAtK().getOrDefault(100, 0.0));
        }

        logger.info("Summary CSV updated: {}", csvFile.toAbsolutePath());
    }

    private void startProgressTracking() {
        logger.info("Starting progress tracking");
        progressScheduler = Executors.newSingleThreadScheduledExecutor();

        // Disabled verbose progress logging - now using Spark's native progress bars
        // progressScheduler.scheduleAtFixedRate(() -> {
        // long progress = progressAccumulator.value();
        // logger.info("Benchmark progress: {}%", Math.min(progress, 100));
        // }, 10, 10, TimeUnit.SECONDS);
    }

    private void stopProgressTracking() {
        if (progressScheduler != null) {
            progressScheduler.shutdown();
            try {
                if (!progressScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    progressScheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                progressScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        logger.info("Progress tracking stopped");
    }

    // Converter methods to unify distributed and single-node result interfaces
    private Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> convertUSearchSparkResults(
            Map<BenchmarkConfig.Precision, USearchSparkBenchmark.BenchmarkResult> sparkResults) {
        Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> converted = new HashMap<>();

        for (Map.Entry<BenchmarkConfig.Precision, USearchSparkBenchmark.BenchmarkResult> entry : sparkResults
                .entrySet()) {
            USearchSparkBenchmark.BenchmarkResult sparkResult = entry.getValue();

            // Convert to single-node result format
            USearchBenchmark.BenchmarkResult singleResult = new USearchBenchmark.BenchmarkResult(
                    sparkResult.getPrecision(), sparkResult.getIndexingTimeMs(), sparkResult.getSearchTimeMs(),
                    sparkResult.getThroughputQPS(), sparkResult.getRecallAtK(), sparkResult.getNDCGAtK(),
                    sparkResult.getMemoryUsageBytes(), sparkResult.getNumVectors(), sparkResult.getDimensions());

            converted.put(entry.getKey(), singleResult);
        }

        return converted;
    }

    private LuceneBenchmark.BenchmarkResult convertLuceneSparkResult(LuceneSparkBenchmark.BenchmarkResult sparkResult) {

        // Convert to single-node result format
        return new LuceneBenchmark.BenchmarkResult(sparkResult.getIndexingTimeMs(), sparkResult.getSearchTimeMs(),
                sparkResult.getPureSearchTimeMs(), sparkResult.getIdRetrievalTimeMs(), sparkResult.getThroughputQPS(),
                sparkResult.getPureSearchQPS(), sparkResult.getRecallAtK(), sparkResult.getNDCGAtK(),
                sparkResult.getMemoryUsageBytes(), sparkResult.getNumVectors(), sparkResult.getDimensions());
    }
}
