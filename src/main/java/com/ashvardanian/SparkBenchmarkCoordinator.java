package com.ashvardanian;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import org.apache.spark.broadcast.Broadcast;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class SparkBenchmarkCoordinator {
    private static final Logger logger = LoggerFactory.getLogger(SparkBenchmarkCoordinator.class);

    public static class BenchmarkResults {
        private final String datasetName;
        private final BenchmarkConfig.BenchmarkMode mode;
        private final long totalTimeMs;
        private final Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults;
        private final LuceneHNSWBenchmark.BenchmarkResult luceneResult;
        private final long timestamp;

        public BenchmarkResults(String datasetName, BenchmarkConfig.BenchmarkMode mode, long totalTimeMs,
                              Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults,
                              LuceneHNSWBenchmark.BenchmarkResult luceneResult) {
            this.datasetName = datasetName;
            this.mode = mode;
            this.totalTimeMs = totalTimeMs;
            this.usearchResults = Collections.unmodifiableMap(new HashMap<>(usearchResults));
            this.luceneResult = luceneResult;
            this.timestamp = System.currentTimeMillis();
        }

        // Getters
        public String getDatasetName() { return datasetName; }
        public BenchmarkConfig.BenchmarkMode getMode() { return mode; }
        public long getTotalTimeMs() { return totalTimeMs; }
        public Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> getUsearchResults() { return usearchResults; }
        public LuceneHNSWBenchmark.BenchmarkResult getLuceneResult() { return luceneResult; }
        public long getTimestamp() { return timestamp; }
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

            // Run benchmarks based on mode
            BenchmarkResults results;
            if (config.getMode() == BenchmarkConfig.BenchmarkMode.LOCAL) {
                results = runLocalBenchmark(config, dataset);
            } else {
                results = runDistributedBenchmark(config, dataset);
            }

            // Save results
            saveResults(config, results);

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

    private BenchmarkResults runLocalBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) 
            throws Exception {
        logger.info("Running local benchmark");

        // Run USearch benchmarks
        progressAccumulator.add(20);
        USearchBenchmark usearchBenchmark = new USearchBenchmark(config, dataset);
        Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults = 
            usearchBenchmark.runBenchmarks();

        progressAccumulator.add(40);

        // Run Lucene benchmark
        LuceneHNSWBenchmark luceneBenchmark = new LuceneHNSWBenchmark(config, dataset);
        LuceneHNSWBenchmark.BenchmarkResult luceneResult = luceneBenchmark.runBenchmark();

        progressAccumulator.add(30);

        long totalTime = System.currentTimeMillis();

        // Log results
        USearchBenchmark.logBenchmarkResults(usearchResults);
        LuceneHNSWBenchmark.logBenchmarkResults(luceneResult);

        return new BenchmarkResults(
            config.getDatasetName(),
            config.getMode(),
            totalTime,
            usearchResults,
            luceneResult
        );
    }

    private BenchmarkResults runDistributedBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) 
            throws Exception {
        logger.info("Running distributed benchmark across Spark cluster");

        // For distributed execution, we would partition the work across Spark nodes
        // This is a simplified implementation that demonstrates the structure
        
        // Broadcast configuration and dataset info to all nodes
        Broadcast<BenchmarkConfig> configBroadcast = spark.sparkContext().broadcast(config, 
            scala.reflect.ClassTag$.MODULE$.apply(BenchmarkConfig.class));
        Broadcast<DatasetRegistry.Dataset> datasetBroadcast = spark.sparkContext().broadcast(dataset,
            scala.reflect.ClassTag$.MODULE$.apply(DatasetRegistry.Dataset.class));

        progressAccumulator.add(20);

        // Create Spark RDD for parallel execution
        // In a full implementation, this would partition the benchmark workload
        List<Integer> partitions = new ArrayList<>();
        int numPartitions = spark.sparkContext().defaultParallelism();
        for (int i = 0; i < numPartitions; i++) {
            partitions.add(i);
        }

        // Distribute benchmark execution
        // This is a simplified version - in practice, you'd partition vectors across nodes
        Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults = 
            runDistributedUSearchBenchmark(configBroadcast, datasetBroadcast, partitions);

        progressAccumulator.add(50);

        LuceneHNSWBenchmark.BenchmarkResult luceneResult = 
            runDistributedLuceneBenchmark(configBroadcast, datasetBroadcast);

        progressAccumulator.add(30);

        // Clean up broadcasts
        configBroadcast.destroy();
        datasetBroadcast.destroy();

        long totalTime = System.currentTimeMillis();

        // Log results
        USearchBenchmark.logBenchmarkResults(usearchResults);
        LuceneHNSWBenchmark.logBenchmarkResults(luceneResult);

        return new BenchmarkResults(
            config.getDatasetName(),
            config.getMode(),
            totalTime,
            usearchResults,
            luceneResult
        );
    }

    private Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> runDistributedUSearchBenchmark(
            Broadcast<BenchmarkConfig> configBroadcast,
            Broadcast<DatasetRegistry.Dataset> datasetBroadcast,
            List<Integer> partitions) throws Exception {
        
        // In a full distributed implementation, this would:
        // 1. Partition the dataset across nodes
        // 2. Run benchmarks on each partition
        // 3. Aggregate results
        
        // For now, we'll run the benchmark locally but demonstrate the distributed structure
        logger.info("Running distributed USearch benchmarks (simplified implementation)");
        
        BenchmarkConfig config = configBroadcast.value();
        DatasetRegistry.Dataset dataset = datasetBroadcast.value();
        
        USearchBenchmark benchmark = new USearchBenchmark(config, dataset);
        return benchmark.runBenchmarks();
    }

    private LuceneHNSWBenchmark.BenchmarkResult runDistributedLuceneBenchmark(
            Broadcast<BenchmarkConfig> configBroadcast,
            Broadcast<DatasetRegistry.Dataset> datasetBroadcast) throws Exception {
        
        logger.info("Running distributed Lucene HNSW benchmark (simplified implementation)");
        
        BenchmarkConfig config = configBroadcast.value();
        DatasetRegistry.Dataset dataset = datasetBroadcast.value();
        
        LuceneHNSWBenchmark benchmark = new LuceneHNSWBenchmark(config, dataset);
        return benchmark.runBenchmark();
    }

    private void saveResults(BenchmarkConfig config, BenchmarkResults results) throws IOException {
        // Create output directory
        Path outputPath = Paths.get(config.getOutputPath());
        Files.createDirectories(outputPath);

        // Save as JSON
        ObjectMapper objectMapper = new ObjectMapper();
        ObjectWriter writer = objectMapper.writerWithDefaultPrettyPrinter();
        
        String filename = String.format("benchmark_results_%s_%s_%d.json", 
                                       results.getDatasetName(), 
                                       results.getMode().getName(),
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
        
        try (FileWriter fw = new FileWriter(csvFile.toFile(), true);
             PrintWriter pw = new PrintWriter(fw)) {
            
            // Write header if file doesn't exist
            if (!fileExists) {
                pw.println("timestamp,dataset,mode,implementation,precision,indexing_time_ms,search_time_ms,throughput_qps,memory_usage_mb,recall_at_1,recall_at_10,recall_at_100");
            }
            
            // Write USearch results
            for (Map.Entry<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> entry : 
                 results.getUsearchResults().entrySet()) {
                
                USearchBenchmark.BenchmarkResult result = entry.getValue();
                pw.printf("%d,%s,%s,USearch,%s,%d,%d,%.2f,%.2f,%.4f,%.4f,%.4f%n",
                    results.getTimestamp(),
                    results.getDatasetName(),
                    results.getMode().getName(),
                    entry.getKey().getName(),
                    result.getIndexingTimeMs(),
                    result.getSearchTimeMs(),
                    result.getThroughputQPS(),
                    result.getMemoryUsageBytes() / (1024.0 * 1024.0),
                    result.getRecallAtK().getOrDefault(1, 0.0),
                    result.getRecallAtK().getOrDefault(10, 0.0),
                    result.getRecallAtK().getOrDefault(100, 0.0)
                );
            }
            
            // Write Lucene result
            LuceneHNSWBenchmark.BenchmarkResult luceneResult = results.getLuceneResult();
            pw.printf("%d,%s,%s,Lucene,F32,%d,%d,%.2f,%.2f,%.4f,%.4f,%.4f%n",
                results.getTimestamp(),
                results.getDatasetName(),
                results.getMode().getName(),
                luceneResult.getIndexingTimeMs(),
                luceneResult.getSearchTimeMs(),
                luceneResult.getThroughputQPS(),
                luceneResult.getMemoryUsageBytes() / (1024.0 * 1024.0),
                luceneResult.getRecallAtK().getOrDefault(1, 0.0),
                luceneResult.getRecallAtK().getOrDefault(10, 0.0),
                luceneResult.getRecallAtK().getOrDefault(100, 0.0)
            );
        }
        
        logger.info("Summary CSV updated: {}", csvFile.toAbsolutePath());
    }

    private void startProgressTracking() {
        logger.info("Starting progress tracking");
        progressScheduler = Executors.newSingleThreadScheduledExecutor();
        
        progressScheduler.scheduleAtFixedRate(() -> {
            long progress = progressAccumulator.value();
            logger.info("Benchmark progress: {}%", Math.min(progress, 100));
        }, 10, 10, TimeUnit.SECONDS);
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
}