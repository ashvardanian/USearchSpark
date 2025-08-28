package com.ashvardanian;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Multithreaded Lucene HNSW benchmark with batch processing
 */
public class LuceneBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(LuceneBenchmark.class);

    public static class SearchMetrics {
        public final Map<Integer, Double> recallAtK;
        public final Map<Integer, Double> ndcgAtK;
        public final long searchTimeMs;

        public SearchMetrics(Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK, long searchTimeMs) {
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.ndcgAtK = Collections.unmodifiableMap(new HashMap<>(ndcgAtK));
            this.searchTimeMs = searchTimeMs;
        }
    }

    public static class BenchmarkResult {
        private final long indexingTimeMs;
        private final long searchTimeMs;
        private final double throughputQPS;
        private final Map<Integer, Double> recallAtK;
        private final Map<Integer, Double> ndcgAtK;
        private final long memoryUsageBytes;
        private final int numVectors;
        private final int dimensions;

        public BenchmarkResult(long indexingTimeMs, long searchTimeMs, double throughputQPS,
                Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK, long memoryUsageBytes,
                int numVectors, int dimensions) {
            this.indexingTimeMs = indexingTimeMs;
            this.searchTimeMs = searchTimeMs;
            this.throughputQPS = throughputQPS;
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.ndcgAtK = Collections.unmodifiableMap(new HashMap<>(ndcgAtK));
            this.memoryUsageBytes = memoryUsageBytes;
            this.numVectors = numVectors;
            this.dimensions = dimensions;
        }

        // Getters
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
    }

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;

    public LuceneBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) {
        this.config = config;
        this.dataset = dataset;
    }

    public BenchmarkResult runBenchmark() throws Exception {
        System.out.println("\n🔍 Starting Lucene HNSW benchmark for dataset: " + dataset.getDefinition().getName());

        // Load base vectors and queries with optional limits
        System.out.print("📂 Loading vectors... ");
        int maxBaseVectors = config.getMaxVectors() > 0 ? (int) Math.min(config.getMaxVectors(), Integer.MAX_VALUE)
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

        System.out.println("📊 Using " + String.format("%,d", numBaseVectors) + " base vectors and " +
                String.format("%,d", queryVectors.getRows()) + " query vectors");

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());

        // Try to load vector IDs for custom document IDs
        BinaryVectorLoader.VectorIds vectorIds = null;
        try {
            String vectorIdsPath = dataset.getVectorIdsPath();
            if (vectorIdsPath != null && !vectorIdsPath.isEmpty()) {
                vectorIds = BinaryVectorLoader.loadVectorIds(vectorIdsPath);
                logger.info("Using custom document IDs from vector ID mapping");
            }
        } catch (Exception e) {
            logger.warn("Could not load vector IDs for indexing: {}", e.getMessage());
        }

        // Create in-memory directory for index
        Directory directory = new ByteBuffersDirectory();

        // Configure index writer with multithreading - use Lucene's default HNSW
        // parameters
        IndexWriterConfig indexConfig = new IndexWriterConfig();
        indexConfig.setUseCompoundFile(false);
        indexConfig.setMaxBufferedDocs(1000);
        indexConfig.setRAMBufferSizeMB(256);

        // Create index writer
        IndexWriter indexWriter = new IndexWriter(directory, indexConfig);

        // Measure indexing time
        long startIndexing = System.currentTimeMillis();
        long memoryBefore = getMemoryUsage();

        // Parallel indexing with thread-safe IndexWriter
        int numThreads = Math.min(java.util.concurrent.ForkJoinPool.commonPool().getParallelism(),
                Math.max(1, numBaseVectors / 1000));
        System.out.println("🧵 Using " + numThreads + " threads for Lucene indexing (" +
                java.util.concurrent.ForkJoinPool.commonPool().getParallelism() + " available)");

        ProgressLogger indexProgress = new ProgressLogger("Indexing F32", numBaseVectors);

        // Parallel indexing - IndexWriter.addDocument() is thread-safe
        java.util.stream.IntStream.range(0, numBaseVectors).parallel().forEach(i -> {
            try {
                // Each thread needs its own buffer to avoid race conditions
                float[] vectorBuffer = new float[baseVectors.getCols()];
                long key = i; // Use index as document ID
                baseVectors.getVectorAsFloat(i, vectorBuffer);

                Document document = new Document();
                document.add(new StoredField("id", key));
                document.add(new KnnFloatVectorField("vector", vectorBuffer)); // No clone needed - buffer is per-thread
                indexWriter.addDocument(document); // Thread-safe operation

                indexProgress.increment();
            } catch (Exception e) {
                throw new RuntimeException("Indexing failed for vector " + i, e);
            }
        });
        indexProgress.complete(numBaseVectors);

        indexWriter.close();

        long indexingTime = System.currentTimeMillis() - startIndexing;
        long memoryAfter = getMemoryUsage();
        // Use Math.max to prevent negative memory usage due to GC
        long memoryUsage = Math.max(0, memoryAfter - memoryBefore);

        // Create index reader and searcher
        DirectoryReader indexReader = DirectoryReader.open(directory);
        IndexSearcher indexSearcher = new IndexSearcher(indexReader);

        // Calculate search metrics (includes both search and accuracy calculation)
        System.out.println("🔍 Searching...");
        SearchMetrics searchMetrics = calculateSearchMetrics(indexSearcher, queryVectors, numQueries,
                config.getKValues());

        // Use the actual search time from metrics (excludes accuracy calculation)
        long searchTime = searchMetrics.searchTimeMs;

        // Calculate throughput based on actual search time
        double throughputQPS = numQueries / (searchTime / 1000.0);

        // Cleanup
        indexReader.close();
        directory.close();

        System.out.println(String.format(
                "✅ Lucene HNSW benchmark completed - Indexing: %,dms, Search: %,dms, Throughput: %,.0f QPS",
                indexingTime, searchTime, throughputQPS));

        return new BenchmarkResult(
                indexingTime,
                searchTime,
                throughputQPS,
                searchMetrics.recallAtK,
                searchMetrics.ndcgAtK,
                memoryUsage,
                numBaseVectors,
                baseVectors.getCols());
    }

    private void addVectorBatchToIndex(IndexWriter indexWriter, VectorProcessor.VectorBatch batch)
            throws IOException {

        String vectorFieldName = "vector";
        List<Document> documents = new ArrayList<>();

        for (int i = 0; i < batch.vectorCount; i++) {
            long key = batch.keys[i];
            float[] vector = new float[batch.dimensions];
            System.arraycopy(batch.vectors, i * batch.dimensions, vector, 0, batch.dimensions);

            Document document = new Document();
            document.add(new StoredField("id", key));
            document.add(new KnnFloatVectorField(vectorFieldName, vector));
            documents.add(document);
        }

        // Lucene IndexWriter is thread-safe for addDocuments() - no sync needed!
        indexWriter.addDocuments(documents);
    }

    private SearchMetrics calculateSearchMetrics(IndexSearcher indexSearcher,
            BinaryVectorLoader.VectorDataset queryVectors,
            int numQueries, int[] kValues) throws Exception {
        String vectorFieldName = "vector";

        // Try to load ground truth if available
        BinaryVectorLoader.GroundTruth groundTruth = null;
        try {
            String groundTruthPath = dataset.getGroundTruthPath();
            if (groundTruthPath != null && !groundTruthPath.isEmpty()) {
                groundTruth = BinaryVectorLoader.loadGroundTruth(groundTruthPath);
                logger.info("Using ground truth for accurate recall calculation");
            }
        } catch (Exception e) {
            logger.warn("Could not load ground truth, using simplified recall calculation: {}", e.getMessage());
        }

        // Try to load vector IDs if available
        BinaryVectorLoader.VectorIds vectorIds = null;
        try {
            String vectorIdsPath = dataset.getVectorIdsPath();
            if (vectorIdsPath != null && !vectorIdsPath.isEmpty()) {
                vectorIds = BinaryVectorLoader.loadVectorIds(vectorIdsPath);
                logger.info("Using vector ID mapping for subset support");
            }
        } catch (Exception e) {
            logger.warn("Could not load vector IDs: {}", e.getMessage());
        }

        // Pre-allocate result arrays to avoid allocation overhead during timing
        TopDocs[] allSearchResults = new TopDocs[numQueries];

        // Find maximum K value for single search optimization
        int maxK = Arrays.stream(kValues).max().orElse(100);

        // Parallel search with IndexSearcher - thread-safe for concurrent reads
        ProgressLogger searchProgress = new ProgressLogger("Searching k=" + maxK, numQueries);
        long startSearch = System.currentTimeMillis();
        java.util.stream.IntStream.range(0, numQueries).parallel().forEach(i -> {
            try {
                // Each thread needs its own buffer to avoid race conditions
                float[] queryBuffer = new float[queryVectors.getCols()];
                queryVectors.getVectorAsFloat(i, queryBuffer);

                // Perform HNSW search with maximum K - IndexSearcher is thread-safe
                KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(vectorFieldName, queryBuffer, maxK);
                allSearchResults[i] = indexSearcher.search(vectorQuery, maxK);

                searchProgress.increment();
            } catch (Exception e) {
                throw new RuntimeException("Search failed for query " + i, e);
            }
        });
        long searchTime = System.currentTimeMillis() - startSearch;
        searchProgress.complete(numQueries);

        // ACCURACY CALCULATION for all K values from single search results (after
        // timing)
        System.out.println("📊 Calculating accuracy metrics...");

        // Initialize result maps for all K values
        Map<Integer, Double> recallResults = new HashMap<>();
        Map<Integer, Double> ndcgResults = new HashMap<>();

        try {
            for (int k : kValues) {
                double totalRecall = 0.0;
                double totalNdcg = 0.0;

                for (int i = 0; i < numQueries; i++) {
                    TopDocs topDocs = allSearchResults[i];

                    // Truncate results to current K (prefix of the max-K search results)
                    int actualK = Math.min(k, topDocs.scoreDocs.length);

                    // Calculate recall using ground truth if available
                    double queryRecall;
                    double queryNdcg = 0.0;
                    if (groundTruth != null && i < groundTruth.getNumQueries()) {
                        // Extract stored document IDs from search results (only first actualK)
                        int[] intResults = new int[actualK];
                        for (int j = 0; j < actualK; j++) {
                            int luceneDocId = topDocs.scoreDocs[j].doc;
                            // Retrieve the stored ID field - this is our consistent [0,N) key
                            org.apache.lucene.document.Document doc = indexSearcher.storedFields()
                                    .document(luceneDocId);
                            long storedId = doc.getField("id").numericValue().longValue();
                            intResults[j] = (int) storedId;
                        }
                        queryRecall = BinaryVectorLoader.calculateRecallAtK(groundTruth, i, intResults, k);
                        queryNdcg = BinaryVectorLoader.calculateNDCGAtK(groundTruth, i, intResults, k);
                    } else {
                        // Simplified recall calculation (no NDCG without ground truth)
                        queryRecall = Math.min(1.0, (double) actualK / k);
                    }

                    totalRecall += queryRecall;
                    totalNdcg += queryNdcg;
                }

                recallResults.put(k, totalRecall / numQueries);
                ndcgResults.put(k, totalNdcg / numQueries);
            }
        } catch (IOException e) {
            throw new RuntimeException("Accuracy calculation failed", e);
        }

        return new SearchMetrics(recallResults, ndcgResults, searchTime);
    }

    private long getMemoryUsage() {
        if (!config.isIncludeMemoryUsage()) {
            return 0;
        }

        Runtime runtime = Runtime.getRuntime();
        runtime.gc();

        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return runtime.totalMemory() - runtime.freeMemory();
    }

    public static void logBenchmarkResults(BenchmarkResult result) {
        System.out.println("=== Lucene HNSW Benchmark Results ===");
        System.out.println("Precision: F32 (baseline)");
        System.out.println(String.format("  Indexing Time: %,d ms", result.getIndexingTimeMs()));
        System.out.println(String.format("  Search Time: %,d ms", result.getSearchTimeMs()));
        System.out.println(String.format("  Indexing Throughput: %,.0f IPS", result.getThroughputIPS()));
        System.out.println(String.format("  Search Throughput: %,.0f QPS", result.getThroughputQPS()));
        System.out.println(
                String.format("  Memory Usage: %,d MB", Math.round(result.getMemoryUsageBytes() / (1024.0 * 1024.0))));

        for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
            int k = entry.getKey();
            double recall = entry.getValue() * 100.0; // Convert to percentage
            double ndcg = result.getNDCGAtK().getOrDefault(k, 0.0) * 100.0; // Convert to percentage
            System.out.println(String.format("  Recall@%d: %.2f%%, NDCG@%d: %.2f%%", k, recall, k, ndcg));
        }

        System.out.println();
    }
}