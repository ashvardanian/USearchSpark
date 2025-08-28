package com.ashvardanian;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;

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
        public final long pureSearchTimeMs;
        public final long idRetrievalTimeMs;
        public final long totalSearchTimeMs;

        public SearchMetrics(Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK,
                long pureSearchTimeMs, long idRetrievalTimeMs) {
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.ndcgAtK = Collections.unmodifiableMap(new HashMap<>(ndcgAtK));
            this.pureSearchTimeMs = pureSearchTimeMs;
            this.idRetrievalTimeMs = idRetrievalTimeMs;
            this.totalSearchTimeMs = pureSearchTimeMs + idRetrievalTimeMs;
        }
    }

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

        public BenchmarkResult(long indexingTimeMs, long searchTimeMs, long pureSearchTimeMs, long idRetrievalTimeMs,
                double throughputQPS, double pureSearchQPS, Map<Integer, Double> recallAtK,
                Map<Integer, Double> ndcgAtK,
                long memoryUsageBytes, int numVectors, int dimensions) {
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

        // Load vector IDs once for consistent ID mapping during indexing
        final BinaryVectorLoader.VectorIds vectorIds = loadVectorIds();

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
        int numThreads = config.getNumThreads() != -1
                ? config.getNumThreads()
                : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();
        System.out.println("🧵 Using " + numThreads + " threads for Lucene indexing (" +
                java.util.concurrent.ForkJoinPool.commonPool().getParallelism() + " available)");

        ProgressLogger indexProgress = new ProgressLogger("Indexing F32", numBaseVectors);

        // Parallel indexing with clean work partitioning
        if (numThreads == 1) {
            // Single-threaded indexing
            float[] vectorBuffer = new float[baseVectors.getCols()];
            for (int i = 0; i < numBaseVectors; i++) {
                baseVectors.getVectorAsFloat(i, vectorBuffer);
                long finalId = (vectorIds != null && i < vectorIds.getNumVectors())
                        ? vectorIds.getId(i)
                        : i;
                Document document = new Document();
                document.add(new StoredField("id", finalId));
                document.add(new KnnFloatVectorField("vector", vectorBuffer));
                indexWriter.addDocument(document);
                indexProgress.increment();
            }
        } else {
            // Multi-threaded indexing with clean work partitioning
            ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
            try {
                List<CompletableFuture<Void>> futures = new ArrayList<>();

                // Partition work evenly across threads
                int vectorsPerThread = numBaseVectors / numThreads;
                int remainingVectors = numBaseVectors % numThreads;

                for (int threadId = 0; threadId < numThreads; threadId++) {
                    final int startIdx = threadId * vectorsPerThread + Math.min(threadId, remainingVectors);
                    final int endIdx = startIdx + vectorsPerThread + (threadId < remainingVectors ? 1 : 0);
                    final int finalThreadId = threadId;

                    CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                        try {
                            float[] vectorBuffer = new float[baseVectors.getCols()];
                            for (int i = startIdx; i < endIdx; i++) {
                                baseVectors.getVectorAsFloat(i, vectorBuffer);
                                long finalId = (vectorIds != null && i < vectorIds.getNumVectors())
                                        ? vectorIds.getId(i)
                                        : i;
                                Document document = new Document();
                                document.add(new StoredField("id", finalId));
                                document.add(new KnnFloatVectorField("vector", vectorBuffer));
                                indexWriter.addDocument(document);
                                indexProgress.increment();
                            }
                        } catch (Exception e) {
                            throw new RuntimeException("Indexing failed in thread " + finalThreadId +
                                    " (range " + startIdx + "-" + endIdx + ")", e);
                        }
                    }, customThreadPool);

                    futures.add(future);
                }

                // Wait for all threads to complete
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

            } finally {
                customThreadPool.shutdown();
            }
        }
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

        // Use the total search time (includes ID retrieval for realistic benchmarking)
        long searchTime = searchMetrics.totalSearchTimeMs;

        // Calculate throughput based on total search time (realistic)
        double throughputQPS = numQueries / (searchTime / 1000.0);

        // Also calculate pure search QPS (theoretical maximum)
        double pureSearchQPS = numQueries / (searchMetrics.pureSearchTimeMs / 1000.0);

        // Cleanup
        indexReader.close();
        directory.close();

        System.out.println(String.format(
                "✅ Lucene HNSW benchmark completed - Indexing: %,dms, Pure Search: %,dms (%,.0f QPS), +ID Retrieval: %,dms (%,.0f QPS)",
                indexingTime, searchMetrics.pureSearchTimeMs, pureSearchQPS, searchMetrics.idRetrievalTimeMs,
                throughputQPS));

        return new BenchmarkResult(
                indexingTime,
                searchTime,
                searchMetrics.pureSearchTimeMs,
                searchMetrics.idRetrievalTimeMs,
                throughputQPS,
                pureSearchQPS,
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

        // IDs are already mapped during indexing - no conversion needed

        // Pre-allocate result arrays to avoid allocation overhead during timing
        long[][] allSearchResults = new long[numQueries][];

        // Find maximum K value for single search optimization
        int maxK = Arrays.stream(kValues).max().orElse(100);

        // Use all available threads for search
        int numThreads = config.getNumThreads() != -1
                ? config.getNumThreads()
                : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();
        System.out.println("🔍 Using " + numThreads + " threads for search");

        // Phase 1: Pure HNSW search (no ID retrieval)
        ProgressLogger pureSearchProgress = new ProgressLogger("Pure search k=" + maxK, numQueries);
        TopDocs[] pureSearchResults = new TopDocs[numQueries];

        long startPureSearch = System.currentTimeMillis();
        if (numThreads == 1) {
            float[] queryBuffer = new float[queryVectors.getCols()];
            for (int i = 0; i < numQueries; i++) {
                queryVectors.getVectorAsFloat(i, queryBuffer);
                KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(vectorFieldName, queryBuffer, maxK);
                pureSearchResults[i] = indexSearcher.search(vectorQuery, maxK);
                pureSearchProgress.increment();
            }
        } else {
            // Multi-threaded search with clean work partitioning
            ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
            try {
                List<CompletableFuture<Void>> futures = new ArrayList<>();

                // Partition work evenly across threads
                int queriesPerThread = numQueries / numThreads;
                int remainingQueries = numQueries % numThreads;

                for (int threadId = 0; threadId < numThreads; threadId++) {
                    final int startIdx = threadId * queriesPerThread + Math.min(threadId, remainingQueries);
                    final int endIdx = startIdx + queriesPerThread + (threadId < remainingQueries ? 1 : 0);
                    final int finalThreadId = threadId;

                    CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                        try {
                            float[] queryBuffer = new float[queryVectors.getCols()];
                            for (int i = startIdx; i < endIdx; i++) {
                                queryVectors.getVectorAsFloat(i, queryBuffer);
                                KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(vectorFieldName, queryBuffer,
                                        maxK);
                                pureSearchResults[i] = indexSearcher.search(vectorQuery, maxK);
                                pureSearchProgress.increment();
                            }
                        } catch (Exception e) {
                            throw new RuntimeException("Pure search failed in thread " + finalThreadId +
                                    " (range " + startIdx + "-" + endIdx + ")", e);
                        }
                    }, customThreadPool);

                    futures.add(future);
                }

                // Wait for all threads to complete
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

            } finally {
                customThreadPool.shutdown();
            }
        }
        long pureSearchTime = System.currentTimeMillis() - startPureSearch;
        pureSearchProgress.complete(numQueries);

        // Phase 2: ID retrieval from search results
        ProgressLogger idRetrievalProgress = new ProgressLogger("ID retrieval k=" + maxK, numQueries);
        long startIdRetrieval = System.currentTimeMillis();

        if (numThreads == 1) {
            for (int i = 0; i < numQueries; i++) {
                allSearchResults[i] = extractDocumentIds(pureSearchResults[i], maxK, indexSearcher);
                idRetrievalProgress.increment();
            }
        } else {
            // Multi-threaded ID retrieval with clean work partitioning
            ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
            try {
                List<CompletableFuture<Void>> futures = new ArrayList<>();

                // Partition work evenly across threads
                int queriesPerThread = numQueries / numThreads;
                int remainingQueries = numQueries % numThreads;

                for (int threadId = 0; threadId < numThreads; threadId++) {
                    final int startIdx = threadId * queriesPerThread + Math.min(threadId, remainingQueries);
                    final int endIdx = startIdx + queriesPerThread + (threadId < remainingQueries ? 1 : 0);
                    final int finalThreadId = threadId;

                    CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                        try {
                            for (int i = startIdx; i < endIdx; i++) {
                                allSearchResults[i] = extractDocumentIds(pureSearchResults[i], maxK, indexSearcher);
                                idRetrievalProgress.increment();
                            }
                        } catch (Exception e) {
                            throw new RuntimeException("ID retrieval failed in thread " + finalThreadId +
                                    " (range " + startIdx + "-" + endIdx + ")", e);
                        }
                    }, customThreadPool);

                    futures.add(future);
                }

                // Wait for all threads to complete
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

            } finally {
                customThreadPool.shutdown();
            }
        }
        long idRetrievalTime = System.currentTimeMillis() - startIdRetrieval;
        idRetrievalProgress.complete(numQueries);

        System.out.println("📊 Calculating accuracy metrics...");

        BinaryVectorLoader.GroundTruth groundTruth = loadGroundTruth();
        Map<Integer, Double> recallResults = new HashMap<>();
        Map<Integer, Double> ndcgResults = new HashMap<>();

        for (int k : kValues) {
            double totalRecall = 0.0;
            double totalNdcg = 0.0;

            for (int i = 0; i < numQueries; i++) {
                long[] docIds = allSearchResults[i];

                // Truncate results to current K (prefix of the max-K search results)
                int actualK = Math.min(k, docIds.length);
                int[] resultsAtK = safeConvertToIntArray(docIds, actualK);

                // Calculate recall using ground truth if available
                double queryRecall;
                double queryNdcg = 0.0;
                if (groundTruth != null && i < groundTruth.getNumQueries()) {
                    queryRecall = BinaryVectorLoader.calculateRecallAtK(groundTruth, i, resultsAtK, k);
                    queryNdcg = BinaryVectorLoader.calculateNDCGAtK(groundTruth, i, resultsAtK, k);
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

        return new SearchMetrics(recallResults, ndcgResults, pureSearchTime, idRetrievalTime);
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

    // Clean helper methods for better separation of concerns

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

    private BinaryVectorLoader.GroundTruth loadGroundTruth() {
        try {
            String groundTruthPath = dataset.getGroundTruthPath();
            if (groundTruthPath != null && !groundTruthPath.isEmpty()) {
                logger.info("Using ground truth for accurate recall calculation");
                return BinaryVectorLoader.loadGroundTruth(groundTruthPath);
            }
        } catch (Exception e) {
            logger.warn("Could not load ground truth, using simplified recall calculation: {}", e.getMessage());
        }
        return null;
    }

    private int[] safeConvertToIntArray(long[] longArray, int length) {
        int[] intArray = new int[length];
        for (int i = 0; i < length; i++) {
            long value = longArray[i];
            if (value > Integer.MAX_VALUE || value < Integer.MIN_VALUE) {
                throw new IllegalArgumentException(
                        "Vector ID " + value + " cannot be safely converted to int at index " + i);
            }
            intArray[i] = (int) value;
        }
        return intArray;
    }

    private long[] extractDocumentIds(TopDocs topDocs, int maxK, IndexSearcher indexSearcher) throws IOException {
        int actualK = Math.min(maxK, topDocs.scoreDocs.length);
        long[] docIds = new long[actualK];

        for (int j = 0; j < actualK; j++) {
            int luceneDocId = topDocs.scoreDocs[j].doc;
            org.apache.lucene.document.Document doc = indexSearcher.storedFields().document(luceneDocId);
            docIds[j] = doc.getField("id").numericValue().longValue();
        }
        return docIds;
    }

    public static void logBenchmarkResults(BenchmarkResult result) {
        System.out.println("=== Lucene HNSW Benchmark Results ===");
        System.out.println("Precision: F32 (baseline)");
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
            double recall = entry.getValue() * 100.0; // Convert to percentage
            double ndcg = result.getNDCGAtK().getOrDefault(k, 0.0) * 100.0; // Convert to percentage
            System.out.println(String.format("  Recall@%d: %.2f%%, NDCG@%d: %.2f%%", k, recall, k, ndcg));
        }

        System.out.println();
    }
}