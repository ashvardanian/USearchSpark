package com.ashvardanian;

import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class USearchSpark {
    private static final Logger logger = LoggerFactory.getLogger(USearchSpark.class);

    public static void main(String[] args) {
        if (args.length < 1) {
            logger.error("Usage: USearchSpark <benchmark_config> [options]");
            logger.error("  benchmark_config: path to benchmark configuration file or dataset name");
            logger.error("  Options:");
            logger.error("    --mode local|cluster");
            logger.error("    --output <output_path>");
            logger.error("    --precision f32,f16,bf16,i8 (comma-separated)");
            System.exit(1);
        }

        SparkSession spark = SparkSession.builder()
                .appName("USearchSpark - Vector Search Benchmark")
                .master("local[*]") // Default to local mode with all available cores
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .getOrCreate();

        try {
            BenchmarkConfig config = BenchmarkConfig.parseArgs(args);
            
            logger.info("Starting USearchSpark Vector Search Benchmark:");
            logger.info("  Dataset: {}", config.getDatasetName());
            logger.info("  Output: {}", config.getOutputPath());
            logger.info("  Mode: {}", config.getMode());
            logger.info("  Precisions: {}", String.join(", ", config.getPrecisionNames()));

            SparkBenchmarkCoordinator coordinator = new SparkBenchmarkCoordinator(spark);
            coordinator.runBenchmark(config);

            logger.info("Benchmark completed successfully");

        } catch (Exception e) {
            logger.error("Benchmark failed", e);
            throw new RuntimeException(e);
        } finally {
            spark.stop();
        }
    }
}