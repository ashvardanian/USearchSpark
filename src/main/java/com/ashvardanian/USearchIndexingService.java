package com.ashvardanian;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

public class USearchIndexingService {
    private static final Logger logger = LoggerFactory.getLogger(USearchIndexingService.class);
    private final SparkSession spark;
    private final ObjectMapper mapper = new ObjectMapper();

    public USearchIndexingService(SparkSession spark) {
        this.spark = spark;
    }

    public void indexShards(Dataset<Row> shardedData, String outputPath, int vectorDimensions) {
        logger.info("Starting USearch indexing with expected {}D vectors", vectorDimensions);

        // Get actual vector dimensions from the first row
        Row firstRow = shardedData.select("embedding").first();
        List<Double> firstEmbedding = firstRow.getList(0);
        int actualVectorDimensions = firstEmbedding.size();

        logger.info("Detected actual vector dimensions: {}", actualVectorDimensions);

        // Get distinct shard IDs
        List<Row> shardIds = shardedData.select("shard_id").distinct().collectAsList();

        logger.info("Processing {} shards for indexing", shardIds.size());

        for (Row shardRow : shardIds) {
            int shardId = shardRow.getInt(0);
            logger.info("Indexing shard {}", shardId);
            indexShard(shardedData, shardId, outputPath + "/shard_" + shardId, actualVectorDimensions);
        }

        saveShardMetadata(shardIds, outputPath);
    }

    private void indexShard(Dataset<Row> shardedData, long shardId, String outputPath, int vectorDimensions) {
        List<Row> shardVectors = shardedData
                .filter(col("shard_id").equalTo(shardId))
                .select("id", "embedding")
                .collectAsList();

        logger.info("Shard {} contains {} vectors", shardId, shardVectors.size());

        if (!shardVectors.isEmpty()) {
            SimpleVectorIndex index = new SimpleVectorIndex(vectorDimensions);

            for (Row row : shardVectors) {
                long id = row.getLong(0);
                List<Double> embeddingList = row.getList(1);
                float[] embedding = new float[embeddingList.size()];
                for (int i = 0; i < embeddingList.size(); i++) {
                    embedding[i] = embeddingList.get(i).floatValue();
                }
                index.addVector(id, embedding);
            }

            index.saveIndex(outputPath);
            logger.info("Saved index for shard {} to {}", shardId, outputPath);
        }
    }

    private void saveShardMetadata(List<Row> shardIds, String outputPath) {
        try {
            Map<String, Object> metadata = new HashMap<>();
            List<Map<String, Object>> shards = new ArrayList<>();

            for (Row row : shardIds) {
                Map<String, Object> shardInfo = new HashMap<>();
                shardInfo.put("shard_id", row.getInt(0));
                shardInfo.put("index_path", "shard_" + row.getInt(0));
                shardInfo.put("created_at", System.currentTimeMillis());
                shards.add(shardInfo);
            }

            metadata.put("shards", shards);
            metadata.put("total_shards", shards.size());

            Files.createDirectories(Paths.get(outputPath));
            String metadataPath = outputPath + "/metadata.json";
            mapper.writerWithDefaultPrettyPrinter().writeValue(new File(metadataPath), metadata);

            logger.info("Saved shard metadata to {}", metadataPath);
        } catch (IOException e) {
            logger.error("Failed to save metadata", e);
        }
    }
}

class SimpleVectorIndex {
    private static final Logger logger = LoggerFactory.getLogger(SimpleVectorIndex.class);
    private final int dimensions;
    private final List<Long> ids = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();

    public SimpleVectorIndex(int dimensions) {
        this.dimensions = dimensions;
    }

    public void addVector(long id, float[] vector) {
        if (vector.length != dimensions) {
            throw new IllegalArgumentException("Vector dimension mismatch");
        }
        ids.add(id);
        vectors.add(vector);
    }

    public void saveIndex(String outputPath) {
        try {
            Files.createDirectories(Paths.get(outputPath));
            String indexPath = outputPath + "/index.bin";

            try (DataOutputStream out = new DataOutputStream(new FileOutputStream(indexPath))) {
                out.writeInt(dimensions);
                out.writeInt(vectors.size());

                for (int i = 0; i < vectors.size(); i++) {
                    out.writeLong(ids.get(i));
                    for (float v : vectors.get(i)) {
                        out.writeFloat(v);
                    }
                }
            }

            logger.info("Saved index with {} vectors to {}", vectors.size(), indexPath);
        } catch (IOException e) {
            logger.error("Failed to save index", e);
            throw new RuntimeException(e);
        }
    }
}