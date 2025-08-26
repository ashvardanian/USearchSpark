package com.ashvardanian;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.spark.sql.functions.*;

public class VectorShardingService {
    private static final Logger logger = LoggerFactory.getLogger(VectorShardingService.class);
    private final SparkSession spark;

    public VectorShardingService(SparkSession spark) {
        this.spark = spark;
    }

    public Dataset<Row> createShards(Dataset<Row> embeddings, long vectorsPerShard, String outputPath) {
        logger.info("Creating shards with max {} vectors per shard", vectorsPerShard);

        long totalVectors = embeddings.count();
        int numShards = (int) Math.ceil((double) totalVectors / vectorsPerShard);

        logger.info("Total vectors: {}, creating {} shards", totalVectors, numShards);

        // Simple hash-based sharding
        Dataset<Row> shardedData = embeddings.withColumn("shard_id", 
            expr("abs(hash(id)) % " + numShards));

        logger.info("Saving sharded data");
        shardedData.write()
                .mode("overwrite")
                .partitionBy("shard_id")
                .parquet(outputPath);

        validateSharding(shardedData, vectorsPerShard);
        return shardedData;
    }

    private void validateSharding(Dataset<Row> shardedData, long maxVectorsPerShard) {
        logger.info("Validating shard distribution");

        Dataset<Row> shardCounts = shardedData.groupBy("shard_id").count();
        shardCounts.show();

        Row stats = shardCounts.agg(
            max("count").alias("max_count"),
            min("count").alias("min_count"),
            avg("count").alias("avg_count")
        ).first();

        logger.info("Sharding statistics:");
        logger.info("  Max vectors per shard: {}", stats.getLong(0));
        logger.info("  Min vectors per shard: {}", stats.getLong(1));
        logger.info("  Avg vectors per shard: {}", stats.getDouble(2));

        if (stats.getLong(0) > maxVectorsPerShard) {
            logger.warn("Some shards exceed maximum size");
        }
    }
}