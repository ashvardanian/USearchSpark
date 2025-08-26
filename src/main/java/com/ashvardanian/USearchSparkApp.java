package com.ashvardanian;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class USearchSparkApp {
    private static final Logger logger = LoggerFactory.getLogger(USearchSparkApp.class);

    public static void main(String[] args) {
        if (args.length < 2) {
            logger.error("Usage: USearchSparkApp <input_path> <output_path> [vector_dimensions] [vectors_per_shard]");
            System.exit(1);
        }

        SparkSession spark = SparkSession.builder()
                .appName("USearchSpark - Large Scale Vector Indexing")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.sql.adaptive.enabled", "true")
                .getOrCreate();

        try {
            String inputPath = args[0];
            String outputPath = args[1];
            int vectorDimensions = args.length > 2 ? Integer.parseInt(args[2]) : 512;
            long vectorsPerShard = args.length > 3 ? Long.parseLong(args[3]) : 1_000_000_000L;

            logger.info("Starting USearchSpark pipeline:");
            logger.info("  Input: {}", inputPath);
            logger.info("  Output: {}", outputPath);
            logger.info("  Vector Dimensions: {}", vectorDimensions);
            logger.info("  Vectors per Shard: {}", vectorsPerShard);

            // Process text and generate embeddings
            TextEmbeddingPipeline pipeline = new TextEmbeddingPipeline(spark);
            Dataset<Row> embeddings = pipeline.processTextData(inputPath, outputPath + "/embeddings", vectorDimensions);

            // Create shards
            VectorShardingService shardingService = new VectorShardingService(spark);
            Dataset<Row> shards = shardingService.createShards(embeddings, vectorsPerShard, outputPath + "/shards");

            // Create USearch indexes
            USearchIndexingService indexingService = new USearchIndexingService(spark);
            indexingService.indexShards(shards, outputPath + "/indexes", vectorDimensions);

            logger.info("Pipeline completed successfully");

        } catch (Exception e) {
            logger.error("Pipeline failed", e);
            throw new RuntimeException(e);
        } finally {
            spark.stop();
        }
    }
}