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
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.MergeScheduler.MergeSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.LRUQueryCache;
import org.apache.lucene.search.UsageTrackingQueryCachingPolicy;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.store.Directory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Multithreaded Lucene HNSW benchmark with batch processing */
public class LuceneBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(LuceneBenchmark.class);

    public static class SearchMetrics {
        public final Map<Integer, Double> recallAtK;
        public final Map<Integer, Double> ndcgAtK;
        public final long pureSearchTimeMs;
        public final long idRetrievalTimeMs;
        public final long totalSearchTimeMs;

        public SearchMetrics(Map<Integer, Double> recallAtK, Map<Integer, Double> ndcgAtK, long pureSearchTimeMs,
                long idRetrievalTimeMs) {
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
                Map<Integer, Double> ndcgAtK, long memoryUsageBytes, int numVectors, int dimensions) {
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

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());

        // Load vector IDs once for consistent ID mapping during indexing
        final BinaryVectorLoader.VectorIds vectorIds = loadVectorIds();

        // Create in-memory directory for index - use our dynamic implementation for
        // large datasets
        Directory directory = new DynamicByteBuffersDirectory();
        System.out.println("📁 Using DynamicByteBuffersDirectory (grows dynamically, no 4GB limit)");

        // Configure index writer and merge policy to keep multiple segments
        IndexWriterConfig indexConfig = new IndexWriterConfig();
        indexConfig.setUseCompoundFile(false);

        // Keep segments reasonably small to enable per-segment parallelism in search
        // and avoid a single huge segment which would serialize HNSW search
        TieredMergePolicy mergePolicy = new TieredMergePolicy();
        mergePolicy.setMaxMergedSegmentMB(1024); // ~1GB max merged segment
        mergePolicy.setSegmentsPerTier(10);
        mergePolicy.setFloorSegmentMB(16);
        indexConfig.setMergePolicy(mergePolicy);

        // Configure HNSW parameters (higher -> better recall, more memory/indexing time)
        indexConfig.setCodec(new CustomHnswCodec(config.getHnswMaxConn(), config.getHnswBeamWidth()));

        // Moderate RAM buffer so we flush into multiple segments
        indexConfig.setRAMBufferSizeMB(512); // 512MB buffer
        indexConfig.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);

        // Configure aggressive concurrent merge scheduler for maximum speed with progress reporting
        int availableCores = Runtime.getRuntime().availableProcessors();
        final java.util.concurrent.atomic.AtomicLong totalMergeTime = new java.util.concurrent.atomic.AtomicLong(0);
        final java.util.concurrent.atomic.AtomicLong totalIndexingStartTime = new java.util.concurrent.atomic.AtomicLong(System.currentTimeMillis());
        
        org.apache.lucene.index.ConcurrentMergeScheduler mergeScheduler = new org.apache.lucene.index.ConcurrentMergeScheduler() {
            @Override
            protected void doMerge(MergeSource mergeSource, org.apache.lucene.index.MergePolicy.OneMerge merge)
                    throws IOException {
                long startMerge = System.currentTimeMillis();
                double sizeMB = merge.estimatedMergeBytes / 1024.0 / 1024.0;
                System.out.println(
                        String.format("🔄 Starting merge of %d segments (%.1f MB)", merge.segments.size(), sizeMB));

                super.doMerge(mergeSource, merge);

                long mergeTime = System.currentTimeMillis() - startMerge;
                totalMergeTime.addAndGet(mergeTime);
                double throughputMBps = sizeMB / (mergeTime / 1000.0);
                
                // Calculate adjusted IPS including merge overhead
                long totalElapsed = System.currentTimeMillis() - totalIndexingStartTime.get();
                double adjustedIPS = totalElapsed > 0 ? (numBaseVectors * 1000.0) / totalElapsed : 0.0;
                
                System.out.println(String.format("✅ Completed merge in %,d ms (%.1f MB/s) - Adjusted IPS: %.0f", 
                    mergeTime, throughputMBps, adjustedIPS));
            }
        };

        // Use ALL available threads for merging to maximize speed
        mergeScheduler.setMaxMergesAndThreads(availableCores * 2, availableCores);
        indexConfig.setMergeScheduler(mergeScheduler);

        // Lucene will handle thread concurrency internally

        System.out.println("🚀 In-memory config: 512MB RAM buffer, tiered merges; HNSW M=" + config.getHnswMaxConn()
                + ", beamWidth=" + config.getHnswBeamWidth() + "; " + availableCores + " available cores");

        // Create index writer
        IndexWriter indexWriter = new IndexWriter(directory, indexConfig);

        // Measure indexing time
        long startIndexing = System.currentTimeMillis();
        totalIndexingStartTime.set(startIndexing); // Update the merge scheduler's start time
        long memoryBefore = getMemoryUsage();

        // Use all available cores for maximum performance
        int numThreads = config.getNumThreads() != -1
                ? config.getNumThreads()
                : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();
        System.out.println("🧵 Using " + numThreads + " threads for Lucene indexing ("
                + java.util.concurrent.ForkJoinPool.commonPool().getParallelism() + " available)");

        ProgressLogger indexProgress = new ProgressLogger("Indexing F32", numBaseVectors);

        // Parallel indexing with clean work partitioning
        if (numThreads == 1) {
            // Single-threaded indexing
            float[] vectorBuffer = new float[baseVectors.getCols()];
            for (int i = 0; i < numBaseVectors; i++) {
                baseVectors.getVectorAsFloat(i, vectorBuffer);
                long finalId = (vectorIds != null && i < vectorIds.getNumVectors()) ? vectorIds.getId(i) : i;
                Document document = new Document();
                document.add(new NumericDocValuesField("id", finalId));
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
                            float[] reusedVector = new float[baseVectors.getCols()]; // Reuse allocation
                            for (int i = startIdx; i < endIdx; i++) {
                                baseVectors.getVectorAsFloat(i, reusedVector);
                                System.arraycopy(reusedVector, 0, vectorBuffer, 0, baseVectors.getCols());
                                long finalId = (vectorIds != null && i < vectorIds.getNumVectors())
                                        ? vectorIds.getId(i)
                                        : i;
                                Document document = new Document();
                                document.add(new NumericDocValuesField("id", finalId));
                                document.add(new KnnFloatVectorField("vector", vectorBuffer));
                                indexWriter.addDocument(document);
                                indexProgress.increment();
                            }
                        } catch (Exception e) {
                            throw new RuntimeException("Indexing failed in thread " + finalThreadId + " (range "
                                    + startIdx + "-" + endIdx + ")", e);
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

        // Commit and close (merging happens automatically with our scheduler)
        indexWriter.commit();
        indexWriter.close();

        long indexingTime = System.currentTimeMillis() - startIndexing;
        long totalMergeTimeValue = totalMergeTime.get();
        long pureIndexingTime = indexingTime - totalMergeTimeValue;
        
        // Report merge statistics
        if (totalMergeTimeValue > 0) {
            double mergeOverheadPercent = (totalMergeTimeValue * 100.0) / indexingTime;
            double pureIPS = pureIndexingTime > 0 ? (numBaseVectors * 1000.0) / pureIndexingTime : 0.0;
            double totalIPS = indexingTime > 0 ? (numBaseVectors * 1000.0) / indexingTime : 0.0;
            System.out.println(String.format("⏱️  Merge overhead: %,d ms (%.1f%%) - Pure IPS: %.0f, Total IPS: %.0f",
                totalMergeTimeValue, mergeOverheadPercent, pureIPS, totalIPS));
        }
        
        long memoryAfter = getMemoryUsage();
        // Use Math.max to prevent negative memory usage due to GC
        long memoryUsageHeap = Math.max(0, memoryAfter - memoryBefore);
        long indexBytes = 0;
        if (directory instanceof DynamicByteBuffersDirectory) {
            indexBytes = ((DynamicByteBuffersDirectory) directory).estimatedSizeInBytes();
        }
        long memoryUsage = Math.max(memoryUsageHeap, indexBytes);

        // Create index reader
        DirectoryReader indexReader = DirectoryReader.open(directory);

        // Report number of segments/leaves for parallelism insight
        int leafCount = indexReader.leaves().size();
        System.out.println("🪵 Index segments (leaves): " + leafCount);

        // Configure IndexSearcher with a balanced thread pool to parallelize
        // per-segment work
        int totalThreads = config.getNumThreads() != -1
                ? config.getNumThreads()
                : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();

        // Per-query parallelism is bounded by number of leaves
        int perQueryThreads = Math.max(1, Math.min(totalThreads, leafCount));

        IndexSearcher indexSearcher;
        if (perQueryThreads > 1) {
            java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors
                    .newFixedThreadPool(perQueryThreads);
            indexSearcher = new IndexSearcher(indexReader, executor);
            System.out.println("🔧 IndexSearcher per-query threads: " + perQueryThreads + " (total desired: "
                    + totalThreads + ")");
        } else {
            indexSearcher = new IndexSearcher(indexReader);
        }

        // Enable query cache for filter-like components (if any)
        indexSearcher.setQueryCache(new LRUQueryCache(4096, 256L * 1024 * 1024)); // 256 MB
        indexSearcher.setQueryCachingPolicy(new UsageTrackingQueryCachingPolicy());

        // Calculate search metrics (includes both search and accuracy calculation)
        System.out.println("🔍 Searching...");
        // Balance outer query concurrency to avoid oversubscription when per-query
        // parallelism is used
        int outerQueryParallelism = Math.max(1, totalThreads / Math.max(1, perQueryThreads));
        SearchMetrics searchMetrics = calculateSearchMetrics(indexSearcher, queryVectors, numQueries,
                config.getKValues(), outerQueryParallelism);

        // Use the total search time (includes ID retrieval for realistic benchmarking)
        long searchTime = searchMetrics.totalSearchTimeMs;

        // Throughputs
        double throughputQPS = numQueries / (searchTime / 1000.0); // total (pure + id retrieval)
        double pureSearchQPS = numQueries / (searchMetrics.pureSearchTimeMs / 1000.0);
        double idRetrievalQPS = searchMetrics.idRetrievalTimeMs > 0
                ? numQueries / (searchMetrics.idRetrievalTimeMs / 1000.0)
                : 0.0;

        // Report memory breakdown
        System.out.println(String.format("🧠 Memory — Index bytes: %,d MB; Heap delta: %,d MB; Reporting: %,d MB",
                Math.round(indexBytes / (1024.0 * 1024.0)), Math.round(memoryUsageHeap / (1024.0 * 1024.0)),
                Math.round(memoryUsage / (1024.0 * 1024.0))));

        // Cleanup
        indexReader.close();
        directory.close();

        System.out.println(String.format(
                "✅ Lucene HNSW benchmark completed - Indexing: %,dms, Pure Search: %,dms (%,.0f QPS), +ID Retrieval: %,dms (%,.0f QPS), Total: %,dms (%,.0f QPS)",
                indexingTime, searchMetrics.pureSearchTimeMs, pureSearchQPS, searchMetrics.idRetrievalTimeMs,
                idRetrievalQPS, searchTime, throughputQPS));

        // Per-core accounting, for clarity
        int indexingThreadsUsed = (config.getNumThreads() != -1)
                ? config.getNumThreads()
                : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();
        int searchThreadsUsed = perQueryThreads * outerQueryParallelism;
        double ipsTotal = numBaseVectors / (indexingTime / 1000.0);
        double ipsPerCore = indexingThreadsUsed > 0 ? ipsTotal / indexingThreadsUsed : 0.0;
        double pureQpsPerCore = searchThreadsUsed > 0 ? pureSearchQPS / searchThreadsUsed : 0.0;
        double idQpsPerCore = searchThreadsUsed > 0 ? idRetrievalQPS / searchThreadsUsed : 0.0;
        double totalQpsPerCore = searchThreadsUsed > 0 ? throughputQPS / searchThreadsUsed : 0.0;

        System.out.println(String.format("   Threads — indexing: %d, search per-query: %d, outer: %d (total: %d)",
                indexingThreadsUsed, perQueryThreads, outerQueryParallelism, searchThreadsUsed));
        System.out.println(String.format(
                "   Per-core — Indexing: %,.0f IPS/core; Pure: %,.2f QPS/core; ID: %,.2f QPS/core; Total: %,.2f QPS/core",
                ipsPerCore, pureQpsPerCore, idQpsPerCore, totalQpsPerCore));

        return new BenchmarkResult(indexingTime, searchTime, searchMetrics.pureSearchTimeMs,
                searchMetrics.idRetrievalTimeMs, throughputQPS, pureSearchQPS, searchMetrics.recallAtK,
                searchMetrics.ndcgAtK, memoryUsage, numBaseVectors, baseVectors.getCols());
    }

    private void addVectorBatchToIndex(IndexWriter indexWriter, VectorProcessor.VectorBatch batch) throws IOException {

        String vectorFieldName = "vector";
        List<Document> documents = new ArrayList<>();

        for (int i = 0; i < batch.vectorCount; i++) {
            long key = batch.keys[i];
            float[] vector = new float[batch.dimensions];
            System.arraycopy(batch.vectors, i * batch.dimensions, vector, 0, batch.dimensions);

            Document document = new Document();
            document.add(new NumericDocValuesField("id", key));
            document.add(new KnnFloatVectorField(vectorFieldName, vector));
            documents.add(document);
        }

        // Lucene IndexWriter is thread-safe for addDocuments() - no sync needed!
        indexWriter.addDocuments(documents);
    }

    private SearchMetrics calculateSearchMetrics(IndexSearcher indexSearcher,
            BinaryVectorLoader.VectorDataset queryVectors, int numQueries, int[] kValues, int outerQueryParallelism)
            throws Exception {
        String vectorFieldName = "vector";

        // IDs are already mapped during indexing - no conversion needed

        // Pre-allocate result arrays to avoid allocation overhead during timing
        long[][] allSearchResults = new long[numQueries][];

        // Find maximum K value for single search optimization
        int maxK = Arrays.stream(kValues).max().orElse(100);

        // Use controlled outer parallelism to avoid oversubscription
        int numThreads = Math.max(1, outerQueryParallelism);
        System.out.println("🔍 Search concurrency — per-query: "
                + Math.max(1,
                        Math.min(indexSearcher.getIndexReader().leaves().size(),
                                config.getNumThreads() != -1
                                        ? config.getNumThreads()
                                        : java.util.concurrent.ForkJoinPool.commonPool().getParallelism()))
                + ", outer: " + numThreads);

        // Phase 1: Pure HNSW search (no ID retrieval)
        ProgressLogger pureSearchProgress = new ProgressLogger("Pure search k=" + maxK, numQueries,
                ProgressLogger.RateUnit.QPS);
        TopDocs[] pureSearchResults = new TopDocs[numQueries];

        long startPureSearch = System.currentTimeMillis();
        if (numThreads == 1) {
            float[] queryBuffer = new float[queryVectors.getCols()];
            for (int i = 0; i < numQueries; i++) {
                queryVectors.getVectorAsFloat(i, queryBuffer);
                KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(vectorFieldName, queryBuffer, maxK);
                // For KNN, IndexSearcher will parallelize across segments when configured
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
                            throw new RuntimeException("Pure search failed in thread " + finalThreadId + " (range "
                                    + startIdx + "-" + endIdx + ")", e);
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
        ProgressLogger idRetrievalProgress = new ProgressLogger("ID retrieval k=" + maxK, numQueries,
                ProgressLogger.RateUnit.QPS);
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
                            throw new RuntimeException("ID retrieval failed in thread " + finalThreadId + " (range "
                                    + startIdx + "-" + endIdx + ")", e);
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
        if (groundTruth == null) {
            System.out.println("⚠️ No ground truth available - accuracy metrics will be zero");
        }

        Map<Integer, Double> recallResults = new HashMap<>();
        Map<Integer, Double> ndcgResults = new HashMap<>();

        for (int k : kValues) {
            double totalRecall = 0.0;
            double totalNdcg = 0.0;
            int validQueries = groundTruth != null ? Math.min(numQueries, groundTruth.getNumQueries()) : 0;

            for (int i = 0; i < validQueries; i++) {
                long[] docIds = allSearchResults[i];
                int actualK = Math.min(k, docIds.length);
                int[] resultsAtK = safeConvertToIntArray(docIds, actualK);

                totalRecall += BinaryVectorLoader.calculateRecallAtK(groundTruth, i, resultsAtK, k);
                totalNdcg += BinaryVectorLoader.calculateNDCGAtK(groundTruth, i, resultsAtK, k);
            }

            recallResults.put(k, validQueries > 0 ? totalRecall / validQueries : 0.0);
            ndcgResults.put(k, validQueries > 0 ? totalNdcg / validQueries : 0.0);
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

        // Use fast DocValues for ID retrieval
        List<LeafReaderContext> leaves = indexSearcher.getIndexReader().leaves();

        for (int j = 0; j < actualK; j++) {
            int luceneDocId = topDocs.scoreDocs[j].doc;
            int leafIndex = ReaderUtil.subIndex(luceneDocId, leaves);
            LeafReaderContext leaf = leaves.get(leafIndex);
            int segDocId = luceneDocId - leaf.docBase;

            NumericDocValues idDv = DocValues.getNumeric(leaf.reader(), "id");
            if (idDv == null || !idDv.advanceExact(segDocId)) {
                throw new IOException("NumericDocValues 'id' missing for docId " + luceneDocId);
            }
            docIds[j] = idDv.longValue();
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
