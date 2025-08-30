package com.ashvardanian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

/** Distributed Spark-based Lucene HNSW benchmark with sharded indexing */
public class LuceneSparkBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(LuceneSparkBenchmark.class);

    public static class BenchmarkResult {
        private final long indexingTimeMs;
        private final long searchTimeMs;
        private final long pureSearchTimeMs;
        private final long idRetrievalTimeMs;
        private final double throughputQPS;
        private final double pureSearchQPS;
        private final Map<Integer, Double> recallAtK;
        private final Map<Integer, Double> ndcgAtK;
        private final long memoryUsageBytes;
        private final int numVectors;
        private final int dimensions;
        private final int numShards;

        public BenchmarkResult(long indexingTimeMs, long searchTimeMs, long pureSearchTimeMs, long idRetrievalTimeMs,
                double throughputQPS, double pureSearchQPS, Map<Integer, Double> recallAtK,
                Map<Integer, Double> ndcgAtK, long memoryUsageBytes, int numVectors, int dimensions, int numShards) {
            this.indexingTimeMs = indexingTimeMs;
            this.searchTimeMs = searchTimeMs;
            this.pureSearchTimeMs = pureSearchTimeMs;
            this.idRetrievalTimeMs = idRetrievalTimeMs;
            this.throughputQPS = throughputQPS;
            this.pureSearchQPS = pureSearchQPS;
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.ndcgAtK = Collections.unmodifiableMap(new HashMap<>(ndcgAtK));
            this.memoryUsageBytes = memoryUsageBytes;
            this.numVectors = numVectors;
            this.dimensions = dimensions;
            this.numShards = numShards;
        }

        // Getters (same as single-node version plus numShards)
        public long getIndexingTimeMs() {
            return indexingTimeMs;
        }

        public long getSearchTimeMs() {
            return searchTimeMs;
        }

        public double getThroughputQPS() {
            return throughputQPS;
        }

        public double getPureSearchQPS() {
            return pureSearchQPS;
        }

        public long getPureSearchTimeMs() {
            return pureSearchTimeMs;
        }

        public long getIdRetrievalTimeMs() {
            return idRetrievalTimeMs;
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
        public final long pureSearchTimeMs;
        public final long idRetrievalTimeMs;

        public SearchMetrics(Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK, long pureSearchTimeMs,
                long idRetrievalTimeMs) {
            this.recallAtK = recallAtK;
            this.ndcgAtK = ndcgAtK;
            this.pureSearchTimeMs = pureSearchTimeMs;
            this.idRetrievalTimeMs = idRetrievalTimeMs;
        }
    }

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;
    private final SparkSession spark;
    private final JavaSparkContext sc;

    public LuceneSparkBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset, SparkSession spark) {
        this.config = config;
        this.dataset = dataset;
        this.spark = spark;
        this.sc = new JavaSparkContext(spark.sparkContext());
    }

    public BenchmarkResult runBenchmark() throws Exception {
        System.out.println(
                "\n🔍 Starting Distributed Lucene HNSW benchmark for dataset: " + dataset.getDefinition().getName());

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

        // Load vector IDs once for consistent ID mapping
        BinaryVectorLoader.VectorIds vectorIds = loadVectorIds();

        System.out.println(String.format(
                "🔧 Creating distributed Lucene HNSW indexes: %,d vectors, %d dims, F32 precision, %d shards",
                numBaseVectors, baseVectors.getCols(), numShards));

        // Create RDD of vector data with shard assignments
        JavaPairRDD<Integer, VectorWithId> vectorRDD = createVectorRDD(baseVectors, numShards, vectorIds);

        // Distributed indexing phase
        long startIndexing = System.currentTimeMillis();
        System.out.println("🔨 Starting distributed indexing across " + numShards + " shards...");

        JavaPairRDD<Integer, SerializableLuceneIndex> indexRDD = vectorRDD.groupByKey().mapToPair(shardData -> {
            int shardId = shardData._1();
            Iterable<VectorWithId> vectors = shardData._2();

            // Create in-memory directory for this shard
            Directory directory = new DynamicByteBuffersDirectory();

            // Configure index writer
            IndexWriterConfig indexConfig = new IndexWriterConfig();
            indexConfig.setUseCompoundFile(false);
            indexConfig.setRAMBufferSizeMB(1024); // 1GB per shard
            indexConfig.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);

            IndexWriter indexWriter = new IndexWriter(directory, indexConfig);

            // Add vectors to shard index
            for (VectorWithId vectorWithId : vectors) {
                Document document = new Document();
                document.add(new StoredField("id", vectorWithId.id));
                document.add(new KnnFloatVectorField("vector", vectorWithId.vector));
                indexWriter.addDocument(document);
            }

            // Commit and close
            indexWriter.commit();
            indexWriter.close();

            return new Tuple2<>(shardId, new SerializableLuceneIndex(directory, shardId));
        });

        // Cache the indexed RDD
        indexRDD.cache();
        long numIndexedShards = indexRDD.count(); // Force computation
        long indexingTime = System.currentTimeMillis() - startIndexing;

        System.out.println(String.format("✅ Distributed indexing completed in %,dms across %d shards", indexingTime,
                numIndexedShards));

        // Distributed search phase
        SearchMetrics metrics = calculateDistributedSearchMetrics(indexRDD, queryVectors, numQueries,
                config.getKValues());

        // Calculate throughput
        long totalSearchTime = metrics.pureSearchTimeMs + metrics.idRetrievalTimeMs;
        double throughputQPS = numQueries / (totalSearchTime / 1000.0);
        double pureSearchQPS = numQueries / (metrics.pureSearchTimeMs / 1000.0);

        // Calculate memory usage (approximate)
        long memoryUsage = estimateDistributedMemoryUsage(numBaseVectors, baseVectors.getCols());

        return new BenchmarkResult(indexingTime, totalSearchTime, metrics.pureSearchTimeMs, metrics.idRetrievalTimeMs,
                throughputQPS, pureSearchQPS, metrics.recallAtK, metrics.ndcgAtK, memoryUsage, numBaseVectors,
                baseVectors.getCols(), (int) numIndexedShards);
    }

    private JavaPairRDD<Integer, VectorWithId> createVectorRDD(BinaryVectorLoader.VectorDataset vectors, int numShards,
            BinaryVectorLoader.VectorIds vectorIds) {

        // Create list of vectors with shard assignments
        List<Tuple2<Integer, VectorWithId>> vectorList = new ArrayList<>();

        for (int i = 0; i < vectors.getRows(); i++) {
            int shardId = i % numShards; // Simple round-robin sharding

            float[] vector = new float[vectors.getCols()];
            vectors.getVectorAsFloat(i, vector);

            long finalId = (vectorIds != null && i < vectorIds.getNumVectors()) ? vectorIds.getId(i) : i;

            vectorList.add(new Tuple2<>(shardId, new VectorWithId(finalId, vector)));
        }

        return sc.parallelizePairs(vectorList, numShards);
    }

    private SearchMetrics calculateDistributedSearchMetrics(JavaPairRDD<Integer, SerializableLuceneIndex> indexRDD,
            BinaryVectorLoader.VectorDataset queryVectors, int numQueries, int[] kValues) throws Exception {

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
            float[] query = new float[queryVectors.getCols()];
            queryVectors.getVectorAsFloat(i, query);
            queryList.add(query);
        }

        Broadcast<List<float[]>> broadcastQueries = sc.broadcast(queryList);
        Broadcast<Integer> broadcastMaxK = sc.broadcast(maxK);

        // Distributed search with scatter-gather
        System.out.println("🔍 Starting distributed search across all shards...");

        // Phase 1: Pure HNSW search
        long startPureSearch = System.currentTimeMillis();
        List<List<SearchResult>> allQueryResults = new ArrayList<>();

        for (int queryId = 0; queryId < numQueries; queryId++) {
            final int finalQueryId = queryId;

            // Search all shards for this query (pure search only)
            JavaRDD<List<SearchResult>> shardResults = indexRDD.map(indexPair -> {
                SerializableLuceneIndex serIndex = indexPair._2();
                float[] query = broadcastQueries.value().get(finalQueryId);

                // Create searcher for this shard
                DirectoryReader reader = DirectoryReader.open(serIndex.directory);
                IndexSearcher searcher = new IndexSearcher(reader);

                // Pure HNSW search
                KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("vector", query, broadcastMaxK.value());
                TopDocs topDocs = searcher.search(vectorQuery, broadcastMaxK.value());

                List<SearchResult> searchResults = new ArrayList<>();
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    searchResults.add(new SearchResult(topDocs.scoreDocs[i].doc, // Store Lucene doc ID temporarily
                            topDocs.scoreDocs[i].score, serIndex.shardId));
                }

                reader.close();
                return searchResults;
            });

            // Collect and merge results from all shards
            List<List<SearchResult>> collectedResults = shardResults.collect();
            List<SearchResult> mergedResults = mergeShardResults(collectedResults, maxK);
            allQueryResults.add(mergedResults);

            if ((queryId + 1) % 100 == 0) {
                System.out.println(String.format("  📊 Pure search processed %d/%d queries", queryId + 1, numQueries));
            }
        }

        long pureSearchTime = System.currentTimeMillis() - startPureSearch;
        System.out.println(String.format("✅ Distributed pure search completed in %,dms", pureSearchTime));

        // Phase 2: ID retrieval from search results
        System.out.println("🔍 Starting distributed ID retrieval...");
        long startIdRetrieval = System.currentTimeMillis();

        List<List<Long>> allQueryIds = new ArrayList<>();

        for (int queryId = 0; queryId < numQueries; queryId++) {
            List<SearchResult> queryResults = allQueryResults.get(queryId);
            final int finalQueryId = queryId;

            // Group results by shard for efficient ID retrieval
            Map<Integer, List<SearchResult>> resultsByShard = new HashMap<>();
            for (SearchResult result : queryResults) {
                resultsByShard.computeIfAbsent(result.shardId, k -> new ArrayList<>()).add(result);
            }

            // Retrieve IDs from each shard in parallel
            JavaRDD<Tuple2<Integer, List<Long>>> shardIdResults = indexRDD.map(indexPair -> {
                int shardId = indexPair._1();
                SerializableLuceneIndex serIndex = indexPair._2();

                List<SearchResult> shardResults = resultsByShard.getOrDefault(shardId, new ArrayList<>());
                List<Long> ids = new ArrayList<>();

                if (!shardResults.isEmpty()) {
                    DirectoryReader reader = DirectoryReader.open(serIndex.directory);
                    IndexSearcher searcher = new IndexSearcher(reader);

                    for (SearchResult result : shardResults) {
                        Document doc = searcher.storedFields().document(result.luceneDocId);
                        long id = doc.getField("id").numericValue().longValue();
                        ids.add(id);
                    }

                    reader.close();
                }

                return new Tuple2<>(shardId, ids);
            });

            // Collect ID results and maintain original order
            List<Tuple2<Integer, List<Long>>> collectedIds = shardIdResults.collect();
            Map<Integer, List<Long>> idsByShard = new HashMap<>();
            for (Tuple2<Integer, List<Long>> entry : collectedIds) {
                idsByShard.put(entry._1(), entry._2());
            }

            // Reconstruct final ID list in score order
            List<Long> finalIds = new ArrayList<>();
            for (SearchResult result : queryResults) {
                List<Long> shardIds = idsByShard.get(result.shardId);
                if (shardIds != null && !shardIds.isEmpty()) {
                    finalIds.add(shardIds.remove(0)); // Take first ID (maintains order within shard)
                }
            }

            allQueryIds.add(finalIds);

            if ((queryId + 1) % 100 == 0) {
                System.out.println(String.format("  📊 ID retrieval processed %d/%d queries", queryId + 1, numQueries));
            }
        }

        long idRetrievalTime = System.currentTimeMillis() - startIdRetrieval;
        System.out.println(String.format("✅ Distributed ID retrieval completed in %,dms", idRetrievalTime));

        // Calculate accuracy metrics
        System.out.println("📊 Calculating distributed accuracy metrics...");
        Map<Integer, Double> recallResults = new HashMap<>();
        Map<Integer, Double> ndcgResults = new HashMap<>();

        for (int k : kValues) {
            double totalRecall = 0.0;
            double totalNdcg = 0.0;
            int validQueries = 0;

            for (int i = 0; i < numQueries; i++) {
                List<Long> queryIds = allQueryIds.get(i);

                // Truncate to k results
                int actualK = Math.min(k, queryIds.size());
                int[] resultsAtK = new int[actualK];
                for (int j = 0; j < actualK; j++) {
                    resultsAtK[j] = queryIds.get(j).intValue();
                }

                double queryRecall = 0.0;
                double queryNdcg = 0.0;
                if (groundTruth != null && i < groundTruth.getNumQueries()) {
                    queryRecall = BinaryVectorLoader.calculateRecallAtK(groundTruth, i, resultsAtK, k);
                    queryNdcg = BinaryVectorLoader.calculateNDCGAtK(groundTruth, i, resultsAtK, k);
                    validQueries++;
                }

                totalRecall += queryRecall;
                totalNdcg += queryNdcg;
            }

            if (validQueries > 0) {
                recallResults.put(k, totalRecall / validQueries);
                ndcgResults.put(k, totalNdcg / validQueries);
            } else {
                recallResults.put(k, 0.0);
                ndcgResults.put(k, 0.0);
            }
        }

        return new SearchMetrics(recallResults, ndcgResults, pureSearchTime, idRetrievalTime);
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
        return (long) numVectors * dimensions * 4 * 3; // 3x for HNSW index overhead
    }

    private BinaryVectorLoader.VectorIds loadVectorIds() {
        try {
            String vectorIdsPath = dataset.getVectorIdsPath();
            if (vectorIdsPath != null && !vectorIdsPath.isEmpty()) {
                logger.info("Using vector ID mapping for subset support");
                return BinaryVectorLoader.loadVectorIds(vectorIdsPath);
            }
        } catch (Exception e) {
            logger.warn("Could not load vector IDs: {}", e.getMessage());
        }
        return null;
    }

    // Helper classes
    private static class SearchResult implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        final int luceneDocId;
        final float score;
        final int shardId;

        SearchResult(int luceneDocId, float score, int shardId) {
            this.luceneDocId = luceneDocId;
            this.score = score;
            this.shardId = shardId;
        }
    }

    private static class VectorWithId implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        final long id;
        final float[] vector;

        VectorWithId(long id, float[] vector) {
            this.id = id;
            this.vector = vector;
        }
    }

    // Wrapper for serializable Lucene index
    private static class SerializableLuceneIndex implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        private final Directory directory;
        private final int shardId;

        SerializableLuceneIndex(Directory directory, int shardId) {
            this.directory = directory;
            this.shardId = shardId;
        }

        public int getShardId() {
            return shardId;
        }
    }

    public static void logBenchmarkResults(BenchmarkResult result) {
        System.out.println("=== Distributed Lucene HNSW Benchmark Results ===");
        System.out.println("Precision: F32 (baseline)");
        System.out.println(String.format("  Shards: %d", result.getNumShards()));
        System.out.println(String.format("  Indexing Time: %,d ms", result.getIndexingTimeMs()));
        System.out.println(String.format("  Pure Search Time: %,d ms", result.getPureSearchTimeMs()));
        System.out.println(String.format("  ID Retrieval Time: %,d ms", result.getIdRetrievalTimeMs()));
        System.out.println(String.format("  Total Search Time: %,d ms", result.getSearchTimeMs()));
        System.out.println(String.format("  Indexing Throughput: %,.0f IPS", result.getThroughputIPS()));
        System.out.println(String.format("  Pure Search Throughput: %,.0f QPS", result.getPureSearchQPS()));
        System.out
                .println(String.format("  Total Search Throughput: %,.0f QPS (realistic)", result.getThroughputQPS()));
        System.out.println(
                String.format("  Memory Usage: %,d MB", Math.round(result.getMemoryUsageBytes() / (1024.0 * 1024.0))));

        for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
            int k = entry.getKey();
            double recall = entry.getValue() * 100.0;
            double ndcg = result.getNDCGAtK().getOrDefault(k, 0.0) * 100.0;
            System.out.println(String.format("  Recall@%d: %.2f%%, NDCG@%d: %.2f%%", k, recall, k, ndcg));
        }

        System.out.println();
    }
}
