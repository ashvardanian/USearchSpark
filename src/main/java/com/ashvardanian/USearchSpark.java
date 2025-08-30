package com.ashvardanian;

import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class USearchSpark {
    private static final Logger logger = LoggerFactory.getLogger(USearchSpark.class);

    public static void main(String[] args) {
        if (args.length < 1 || "--help".equals(args[0]) || "-h".equals(args[0])) {
            printHelp();
            System.exit(args.length < 1 ? 1 : 0);
        }

        SparkSession spark = null;
        try {
            BenchmarkConfig config = BenchmarkConfig.parseArgs(args);

            spark =
                    SparkSession.builder()
                            .appName("USearchSpark - Vector Search Benchmark")
                            .master("local[*]") // Use all available cores
                            .config(
                                    "spark.serializer",
                                    "org.apache.spark.serializer.KryoSerializer")
                            .config("spark.sql.adaptive.enabled", "true")
                            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                            .getOrCreate();

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
            if (spark != null) {
                spark.stop();
            }
        }
    }

    private static void printHelp() {
        String help =
                """
                USearchSpark - Vector Search Benchmark (USearch vs Lucene HNSW)

                USAGE
                  gradle run --args="<dataset> [options]"

                DATASETS
                  yandex-t2i-1m       1M vectors, 200 dims, float32
                  yandex-deep-1m      1M vectors, 96 dims, float32
                  msft-spacev-100m    100M vectors, 100 dims, int8 (9.3GB)
                  msft-spacev-1b      1B vectors, 100 dims, int8 (131GB)

                OPTIONS
                  --max-vectors N        Limit vectors to index (default: all, use -1 for all)
                  --precision LIST       Comma-separated: f32,f16,bf16,i8 (default: all)
                  --output PATH          Results directory (default: stats)
                  --queries N            Number of queries (default: 10000)
                  --k-values LIST        Comma-separated k values (default: 1,10,100)
                  --batch-size N         Parallel processing batch size (default: 1024)
                  --threads N            Number of threads (default: hardware threads)
                  --help, -h             Show this help

                EXAMPLES
                  # Quick test with 100K vectors
                  gradle run --args="msft-spacev-100m --max-vectors 100000"

                  # Test only INT8 precision
                  gradle run --args="yandex-t2i-1m --precision i8"

                  # Full dataset on large machine
                  gradle run --args="msft-spacev-100m --max-vectors -1"

                NOTE: Large datasets require significant memory. Use --max-vectors to limit.
                """;
        System.out.print(help);
    }
}
