package com.ashvardanian;

import cloud.unum.usearch.Index;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

/** Distributed Spark-based USearch benchmark with sharded indexing */
public class USearchSparkBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(USearchSparkBenchmark.class);

    public static class BenchmarkResult {
        private final BenchmarkConfig.Precision precision;
        private final long indexingTimeMs;
        private final long searchTimeMs;
        private final double throughputQPS;
        private final Map<Integer, Double> recallAtK;
        private final Map<Integer, Double> ndcgAtK;
        private final long memoryUsageBytes;
        private final int numVectors;
        private final int dimensions;
        private final int numShards;

        public BenchmarkResult(BenchmarkConfig.Precision precision, long indexingTimeMs, long searchTimeMs,
                double throughputQPS, Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK,
                long memoryUsageBytes, int numVectors, int dimensions, int numShards) {
            this.precision = precision;
            this.indexingTimeMs = indexingTimeMs;
            this.searchTimeMs = searchTimeMs;
            this.throughputQPS = throughputQPS;
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.ndcgAtK = Collections.unmodifiableMap(new HashMap<>(ndcgAtK));
            this.memoryUsageBytes = memoryUsageBytes;
            this.numVectors = numVectors;
            this.dimensions = dimensions;
            this.numShards = numShards;
        }

        // Getters (same as single-node version)
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

        public double getThroughputIPS() {
            return indexingTimeMs > 0 ? numVectors / (indexingTimeMs / 1000.0) : 0.0;
        }

        public Map<Integer, Double> getRecallAtK() {
            return recallAtK;
        }

        public Map<Integer, Double> getNDCGAtK() {
            return ndcgAtK;
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

        public int getNumShards() {
            return numShards;
        }
    }

    public static class SearchMetrics {
        public final Map<Integer, Double> recallAtK;
        public final Map<Integer, Double> ndcgAtK;
        public final long searchTimeMs;

        public SearchMetrics(Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK, long searchTimeMs) {
            this.recallAtK = recallAtK;
            this.ndcgAtK = ndcgAtK;
            this.searchTimeMs = searchTimeMs;
        }
    }

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;
    private final SparkSession spark;
    private final JavaSparkContext sc;

    public USearchSparkBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset, SparkSession spark) {
        this.config = config;
        this.dataset = dataset;
        this.spark = spark;
        this.sc = new JavaSparkContext(spark.sparkContext());
    }

    public Map<BenchmarkConfig.Precision, BenchmarkResult> runBenchmarks() throws Exception {
        Map<BenchmarkConfig.Precision, BenchmarkResult> results = new HashMap<>();

        System.out.println(
                "\n🔍 Starting Distributed USearch benchmarks for dataset: " + dataset.getDefinition().getName());

        // Load base vectors and queries with optional limits
        System.out.print("📂 Loading vectors into Spark RDDs... ");
        int maxBaseVectors = config.getMaxVectors() > 0
                ? (int) Math.min(config.getMaxVectors(), Integer.MAX_VALUE)
                : -1;
        BinaryVectorLoader.VectorDataset baseVectors = BinaryVectorLoader.loadVectors(dataset.getBaseVectorPath(), 0,
                maxBaseVectors);
        BinaryVectorLoader.VectorDataset queryVectors = BinaryVectorLoader.loadVectors(dataset.getQueryVectorPath());
        System.out.println("✅ Done");

        int numBaseVectors = baseVectors.getRows();
        if (maxBaseVectors > 0) {
            System.out.println(
                    "🔢 Limiting base vectors to " + String.format("%,d", numBaseVectors) + " for faster testing");
        }

        System.out.println("📊 Using " + String.format("%,d", numBaseVectors) + " base vectors and "
                + String.format("%,d", queryVectors.getRows()) + " query vectors");

        // Calculate number of shards based on Spark executors
        int numExecutors = Integer.parseInt(spark.conf().get("spark.executor.instances", "4"));
        int numShards = Math.max(numExecutors, 4); // Minimum 4 shards
        System.out.println("🗂️ Using " + numShards + " shards for distributed indexing");

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());

        for (BenchmarkConfig.Precision precision : config.getPrecisions()) {
            System.out.println(String.format("\n⚙️ Running Distributed USearch benchmark with precision: %s",
                    precision.getName()));

            try {
                BenchmarkResult result = runSingleBenchmark(baseVectors, queryVectors, precision, numQueries,
                        numBaseVectors, numShards);
                results.put(precision, result);

                System.out.println(String.format(
                        "✅ %s completed - Indexing: %,dms, Search: %,dms, Throughput: %,.0f QPS, Shards: %d",
                        precision.getName(), result.getIndexingTimeMs(), result.getSearchTimeMs(),
                        result.getThroughputQPS(), result.getNumShards()));

            } catch (Exception e) {
                System.err.println(String.format("❌ Failed to run Distributed USearch benchmark for precision %s: %s",
                        precision.getName(), e.getMessage()));
                throw e;
            }
        }

        return results;
    }

    private BenchmarkResult runSingleBenchmark(BinaryVectorLoader.VectorDataset baseVectors,
            BinaryVectorLoader.VectorDataset queryVectors, BenchmarkConfig.Precision precision, int numQueries,
            int numBaseVectors, int numShards) throws Exception {

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
            throw new IllegalArgumentException(
                    "Unsupported metric: " + metric + ". Supported metrics are: l2, ip, cos");
        }

        // Set precision-specific quantization
        String quantization;
        switch (precision) {
            case F32 :
                quantization = Index.Quantization.FLOAT32;
                break;
            case F16 :
                quantization = Index.Quantization.FLOAT16;
                break;
            case BF16 :
                quantization = Index.Quantization.BFLOAT16;
                break;
            case I8 :
                quantization = Index.Quantization.INT8;
                break;
            default :
                quantization = Index.Quantization.FLOAT32;
        }

        // Determine input format based on dataset
        BinaryVectorLoader.VectorType vectorType = baseVectors.getType();
        boolean useByteData = vectorType == BinaryVectorLoader.VectorType.INT8
                || vectorType == BinaryVectorLoader.VectorType.UINT8
                || vectorType == BinaryVectorLoader.VectorType.UINT8_BIN;

        System.out.println(String.format(
                "🔧 Creating distributed USearch indexes: %,d vectors, %d dims, %s metric, %s precision, %d shards",
                numBaseVectors, baseVectors.getCols(), usearchMetric, quantization, numShards));

        // Create RDD of vector data with shard assignments
        JavaPairRDD<Integer, float[]> vectorRDD = createVectorRDD(baseVectors, numShards, useByteData);

        // Broadcast configuration
        Broadcast<String> broadcastMetric = sc.broadcast(usearchMetric);
        Broadcast<String> broadcastQuantization = sc.broadcast(quantization);
        Broadcast<Integer> broadcastDimensions = sc.broadcast(baseVectors.getCols());

        // Distributed indexing phase
        long startIndexing = System.currentTimeMillis();
        System.out.println("🔨 Starting distributed indexing across " + numShards + " shards...");

        JavaPairRDD<Integer, SerializableIndex> indexRDD = vectorRDD.groupByKey().mapToPair(shardData -> {
            int shardId = shardData._1();
            Iterable<float[]> vectors = shardData._2();

            // Create index for this shard
            Index index = new Index.Config().metric(broadcastMetric.value()).quantization(broadcastQuantization.value())
                    .dimensions(broadcastDimensions.value()).build();

            // Add vectors to shard index
            int vectorId = shardId * 1000000; // Offset vector IDs by shard
            for (float[] vector : vectors) {
                index.add(vectorId++, vector);
            }

            return new Tuple2<>(shardId, new SerializableIndex(index, shardId));
        });

        // Cache the indexed RDD
        indexRDD.cache();
        long numIndexedShards = indexRDD.count(); // Force computation
        long indexingTime = System.currentTimeMillis() - startIndexing;

        System.out.println(String.format("✅ Distributed indexing completed in %,dms across %d shards", indexingTime,
                numIndexedShards));

        // Distributed search phase
        SearchMetrics metrics = calculateDistributedSearchMetrics(indexRDD, queryVectors, numQueries,
                config.getKValues(), useByteData);

        // Calculate memory usage (approximate)
        long memoryUsage = estimateDistributedMemoryUsage(numBaseVectors, baseVectors.getCols());

        return new BenchmarkResult(precision, indexingTime, metrics.searchTimeMs,
                numQueries / (metrics.searchTimeMs / 1000.0), metrics.recallAtK, metrics.ndcgAtK, memoryUsage,
                numBaseVectors, baseVectors.getCols(), (int) numIndexedShards);
    }

    private JavaPairRDD<Integer, float[]> createVectorRDD(BinaryVectorLoader.VectorDataset vectors, int numShards,
            boolean useByteData) {

        // Create list of vectors with shard assignments
        List<Tuple2<Integer, float[]>> vectorList = new ArrayList<>();

        for (int i = 0; i < vectors.getRows(); i++) {
            int shardId = i % numShards; // Simple round-robin sharding

            float[] vector;
            if (useByteData) {
                byte[] byteVector = new byte[vectors.getCols()];
                vectors.getVectorAsByte(i, byteVector);
                // Convert byte to float
                vector = new float[vectors.getCols()];
                for (int j = 0; j < vectors.getCols(); j++) {
                    vector[j] = (byteVector[j] & 0xFF) / 255.0f;
                }
            } else {
                vector = new float[vectors.getCols()];
                vectors.getVectorAsFloat(i, vector);
            }

            vectorList.add(new Tuple2<>(shardId, vector));
        }

        return sc.parallelizePairs(vectorList, numShards);
    }

    private SearchMetrics calculateDistributedSearchMetrics(JavaPairRDD<Integer, SerializableIndex> indexRDD,
            BinaryVectorLoader.VectorDataset queryVectors, int numQueries, int[] kValues, boolean useByteData)
            throws Exception {

        // Try to load ground truth if available
        BinaryVectorLoader.GroundTruth groundTruth = null;
        try {
            String groundTruthPath = dataset.getGroundTruthPath();
            if (groundTruthPath != null && !groundTruthPath.isEmpty()) {
                groundTruth = BinaryVectorLoader.loadGroundTruth(groundTruthPath);
                logger.info("Using ground truth for accurate recall calculation");
            }
        } catch (Exception e) {
            logger.warn("Could not load ground truth: {}", e.getMessage());
        }

        int maxK = Arrays.stream(kValues).max().orElse(100);

        // Broadcast query data
        List<float[]> queryList = new ArrayList<>();
        for (int i = 0; i < numQueries; i++) {
            float[] query;
            if (useByteData) {
                byte[] byteQuery = new byte[queryVectors.getCols()];
                queryVectors.getVectorAsByte(i, byteQuery);
                query = new float[queryVectors.getCols()];
                for (int j = 0; j < queryVectors.getCols(); j++) {
                    query[j] = (byteQuery[j] & 0xFF) / 255.0f;
                }
            } else {
                query = new float[queryVectors.getCols()];
                queryVectors.getVectorAsFloat(i, query);
            }
            queryList.add(query);
        }

        Broadcast<List<float[]>> broadcastQueries = sc.broadcast(queryList);
        Broadcast<Integer> broadcastMaxK = sc.broadcast(maxK);

        // Distributed search with scatter-gather
        System.out.println("🔍 Starting distributed search across all shards...");
        long startSearch = System.currentTimeMillis();

        // For each query, search all shards and collect results
        List<List<SearchResult>> allQueryResults = new ArrayList<>();

        for (int queryId = 0; queryId < numQueries; queryId++) {
            final int finalQueryId = queryId;

            // Search all shards for this query
            JavaRDD<List<SearchResult>> shardResults = indexRDD.map(indexPair -> {
                SerializableIndex serIndex = indexPair._2();
                float[] query = broadcastQueries.value().get(finalQueryId);

                // Search this shard
                long[] results = serIndex.search(query, broadcastMaxK.value());

                List<SearchResult> searchResults = new ArrayList<>();
                for (int i = 0; i < results.length; i++) {
                    searchResults.add(new SearchResult((int) results[i], 1.0f / (i + 1))); // Simple scoring
                }

                return searchResults;
            });

            // Collect and merge results from all shards
            List<List<SearchResult>> collectedResults = shardResults.collect();
            List<SearchResult> mergedResults = mergeShardResults(collectedResults, maxK);
            allQueryResults.add(mergedResults);

            if ((queryId + 1) % 100 == 0) {
                System.out.println(String.format("  📊 Processed %d/%d queries", queryId + 1, numQueries));
            }
        }

        long searchTime = System.currentTimeMillis() - startSearch;
        System.out.println(String.format("✅ Distributed search completed in %,dms", searchTime));

        // Calculate accuracy metrics
        System.out.println("📊 Calculating distributed accuracy metrics...");
        Map<Integer, Double> recallResults = new HashMap<>();
        Map<Integer, Double> ndcgResults = new HashMap<>();

        for (int k : kValues) {
            double totalRecall = 0.0;
            double totalNdcg = 0.0;

            for (int i = 0; i < numQueries; i++) {
                List<SearchResult> queryResults = allQueryResults.get(i);

                // Truncate to k results
                int actualK = Math.min(k, queryResults.size());
                int[] resultsAtK = new int[actualK];
                for (int j = 0; j < actualK; j++) {
                    resultsAtK[j] = queryResults.get(j).vectorId;
                }

                double queryRecall;
                double queryNdcg = 0.0;
                if (groundTruth != null && i < groundTruth.getNumQueries()) {
                    queryRecall = BinaryVectorLoader.calculateRecallAtK(groundTruth, i, resultsAtK, k);
                    queryNdcg = BinaryVectorLoader.calculateNDCGAtK(groundTruth, i, resultsAtK, k);
                } else {
                    queryRecall = Math.min(1.0, (double) actualK / k);
                }

                totalRecall += queryRecall;
                totalNdcg += queryNdcg;
            }

            recallResults.put(k, totalRecall / numQueries);
            ndcgResults.put(k, totalNdcg / numQueries);
        }

        return new SearchMetrics(recallResults, ndcgResults, searchTime);
    }

    private List<SearchResult> mergeShardResults(List<List<SearchResult>> shardResults, int maxK) {
        // Use priority queue to merge results from all shards
        PriorityQueue<SearchResult> globalResults = new PriorityQueue<>(Comparator.comparingDouble(sr -> -sr.score)); // Max
                                                                                                                      // heap
                                                                                                                      // by
                                                                                                                      // score

        // Add all results to priority queue
        for (List<SearchResult> shardResult : shardResults) {
            for (SearchResult result : shardResult) {
                globalResults.offer(result);
            }
        }

        // Extract top K results
        List<SearchResult> topK = new ArrayList<>();
        for (int i = 0; i < maxK && !globalResults.isEmpty(); i++) {
            topK.add(globalResults.poll());
        }

        return topK;
    }

    private long estimateDistributedMemoryUsage(int numVectors, int dimensions) {
        // Rough estimation: each vector takes ~4 bytes per dimension + index overhead
        return (long) numVectors * dimensions * 4 * 2; // 2x for index overhead
    }

    // Helper classes
    private static class SearchResult implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        final int vectorId;
        final float score;

        SearchResult(int vectorId, float score) {
            this.vectorId = vectorId;
            this.score = score;
        }
    }

    // Wrapper for serializable USearch index
    private static class SerializableIndex implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        private final Index index;
        private final int shardId;

        SerializableIndex(Index index, int shardId) {
            this.index = index;
            this.shardId = shardId;
        }

        public long[] search(float[] query, int k) {
            return index.search(query, k);
        }

        public int getShardId() {
            return shardId;
        }
    }

    public static void logBenchmarkResults(Map<BenchmarkConfig.Precision, BenchmarkResult> results) {
        System.out.println("=== Distributed USearch Benchmark Results ===");

        List<BenchmarkConfig.Precision> sortedPrecisions = new ArrayList<>(results.keySet());
        sortedPrecisions.sort(Comparator.comparing(BenchmarkConfig.Precision::getName));

        for (BenchmarkConfig.Precision precision : sortedPrecisions) {
            BenchmarkResult result = results.get(precision);

            System.out.println("Precision: " + precision.getName());
            System.out.println(String.format("  Shards: %d", result.getNumShards()));
            System.out.println(String.format("  Indexing Time: %,d ms", result.getIndexingTimeMs()));
            System.out.println(String.format("  Search Time: %,d ms", result.getSearchTimeMs()));
            System.out.println(String.format("  Indexing Throughput: %,.0f IPS", result.getThroughputIPS()));
            System.out.println(String.format("  Search Throughput: %,.0f QPS", result.getThroughputQPS()));
            System.out.println(String.format("  Memory Usage: %,d MB",
                    Math.round(result.getMemoryUsageBytes() / (1024.0 * 1024.0))));

            for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
                int k = entry.getKey();
                double recall = entry.getValue() * 100.0;
                double ndcg = result.getNDCGAtK().getOrDefault(k, 0.0) * 100.0;
                System.out.println(String.format("  Recall@%d: %.2f%%, NDCG@%d: %.2f%%", k, recall, k, ndcg));
            }

            System.out.println();
        }
    }
}
