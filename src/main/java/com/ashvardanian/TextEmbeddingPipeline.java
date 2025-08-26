package com.ashvardanian;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.spark.sql.functions.*;

public class TextEmbeddingPipeline {
    private static final Logger logger = LoggerFactory.getLogger(TextEmbeddingPipeline.class);
    private final SparkSession spark;

    public TextEmbeddingPipeline(SparkSession spark) {
        this.spark = spark;
    }

    public Dataset<Row> processTextData(String inputPath, String outputPath, int vectorDimensions) {
        logger.info("Starting text processing pipeline from {}", inputPath);

        Dataset<Row> rawData = loadTextData(inputPath);
        Dataset<Row> processedData = preprocessText(rawData);
        Dataset<Row> embeddings = generateEmbeddings(processedData, vectorDimensions);

        logger.info("Saving embeddings to {}", outputPath);
        embeddings.write()
                .mode("overwrite")
                .parquet(outputPath);

        return embeddings;
    }

    private Dataset<Row> loadTextData(String inputPath) {
        logger.info("Loading text data");
        return spark.read()
                .option("multiline", "true")
                .option("encoding", "UTF-8")
                .text(inputPath)
                .withColumn("id", monotonically_increasing_id())
                .select(col("id"), col("value").as("text"))
                .filter(length(col("text")).gt(10));
    }

    private Dataset<Row> preprocessText(Dataset<Row> df) {
        logger.info("Preprocessing text data");

        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("text")
                .setOutputCol("raw_tokens")
                .setPattern("\\W")
                .setToLowercase(true);

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("raw_tokens")
                .setOutputCol("tokens");

        Pipeline pipeline = new Pipeline()
                .setStages(new org.apache.spark.ml.PipelineStage[]{tokenizer, stopWordsRemover});

        return pipeline.fit(df).transform(df)
                .filter(size(col("tokens")).gt(0))
                .select(col("id"), col("text"), col("tokens"));
    }

    private Dataset<Row> generateEmbeddings(Dataset<Row> df, int vectorDimensions) {
        logger.info("Generating embeddings with dimension {}", vectorDimensions);

        HashingTF hashingTF = new HashingTF()
                .setInputCol("tokens")
                .setOutputCol("raw_features")
                .setNumFeatures(vectorDimensions * 2);

        IDF idf = new IDF()
                .setInputCol("raw_features")
                .setOutputCol("features")
                .setMinDocFreq(2);

        Pipeline pipeline = new Pipeline()
                .setStages(new org.apache.spark.ml.PipelineStage[]{hashingTF, idf});

        Dataset<Row> result = pipeline.fit(df).transform(df);

        // Convert Vector to array
        return result.select(
                col("id"),
                col("text"),
                udf((Vector v) -> v.toArray(), org.apache.spark.sql.types.DataTypes.createArrayType(
                        org.apache.spark.sql.types.DataTypes.DoubleType))
                        .apply(col("features")).as("embedding")
        );
    }
}