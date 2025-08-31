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
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.MergeScheduler.MergeSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.KnnByteVectorQuery;
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
    private final BenchmarkConfig.Precision precision;

    public LuceneBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset,
            BenchmarkConfig.Precision precision) {
        this.config = config;
        this.dataset = dataset;
        this.precision = precision;
    }

    /**
     * Monitors memory pressure and triggers GC if needed to prevent OOM errors. Called periodically during indexing to
     * ensure stable performance.
     */
    private void checkMemoryPressure(IndexWriter indexWriter, long gcThresholdBytes, int vectorIndex) {
        if (vectorIndex % 1000 != 0) {
            return; // Only check every 1000 vectors
        }

        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();

        if (usedMemory > gcThresholdBytes) {
            System.out.println(String.format("⚠️  Memory pressure detected (%,d MB used), pausing for GC...",
                    usedMemory / (1024 * 1024)));
            try {
                indexWriter.commit(); // Flush pending changes
                System.gc(); // Request GC
                Thread.sleep(500); // Give GC time to work

                // Log memory after GC
                long newUsedMemory = runtime.totalMemory() - runtime.freeMemory();
                System.out.println(
                        String.format("✅ GC completed (%,d MB freed)", (usedMemory - newUsedMemory) / (1024 * 1024)));
            } catch (Exception e) {
                // Log but don't fail - memory management is best-effort
                System.err.println("Warning: Failed to manage memory pressure: " + e.getMessage());
            }
        }
    }

    public BenchmarkResult runBenchmark() throws Exception {
        System.out.println("\n🔍 Starting Lucene HNSW benchmark for dataset: " + dataset.getDefinition().getName()
                + " with precision: " + precision.getName());

        // Load base vectors and queries with optional limits
        System.out.print("📂 Loading vectors... ");
        int maxBaseVectors = config.getMaxVectors() > 0 ? (int)Math.min(config.getMaxVectors(), Integer.MAX_VALUE) : -1;
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

        // Fixed 4GB segment size for predictable performance

        // Estimate index size: vector data + HNSW overhead (~2.5x for graph structure)
        long vectorBytes = (long)numBaseVectors * baseVectors.getCols() * 4; // f32 = 4 bytes per float
        long estimatedIndexSizeBytes = (long)(vectorBytes * 2.5); // HNSW overhead factor
        long estimatedIndexSizeMB = estimatedIndexSizeBytes / (1024 * 1024);

        // Fixed 4GB segments
        int optimalSegmentSizeMB = 4096; // 4GB segments
        long estimatedSegments = estimatedIndexSizeMB / optimalSegmentSizeMB;

        System.out.println(String.format("📊 Target segment size: 4GB (expecting ~%d segments for %,d MB index)",
                estimatedSegments, estimatedIndexSizeMB));

        TieredMergePolicy mergePolicy = new TieredMergePolicy();
        mergePolicy.setMaxMergedSegmentMB(optimalSegmentSizeMB);
        mergePolicy.setFloorSegmentMB(256); // Reasonable floor
        indexConfig.setMergePolicy(mergePolicy);

        // Configure HNSW parameters (higher -> better recall, more memory/indexing time)
        indexConfig.setCodec(new CustomHnswCodec(config.getHnswMaxConn(), config.getHnswBeamWidth()));

        // Moderate RAM buffer so we flush into multiple segments
        indexConfig.setRAMBufferSizeMB(512); // 512MB buffer
        indexConfig.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);

        // Configure aggressive concurrent merge scheduler for maximum speed with progress reporting
        final java.util.concurrent.atomic.AtomicLong totalMergeTime = new java.util.concurrent.atomic.AtomicLong(0);
        final java.util.concurrent.atomic.AtomicLong totalIndexingStartTime = new java.util.concurrent.atomic.AtomicLong(
                System.currentTimeMillis());
        final java.util.concurrent.atomic.AtomicInteger mergeCount = new java.util.concurrent.atomic.AtomicInteger(0);

        org.apache.lucene.index.ConcurrentMergeScheduler mergeScheduler = new org.apache.lucene.index.ConcurrentMergeScheduler() {
            @Override
            protected void doMerge(MergeSource mergeSource, org.apache.lucene.index.MergePolicy.OneMerge merge)
                    throws IOException {
                long startMerge = System.currentTimeMillis();
                double sizeMB = merge.estimatedMergeBytes / 1024.0 / 1024.0;

                // Only log significant merges (>50MB) or every 20th merge to reduce noise
                int currentMergeCount = mergeCount.incrementAndGet();
                boolean shouldLog = sizeMB > 50.0 || (currentMergeCount % 20 == 0);

                if (shouldLog) {
                    System.out.println(String.format("🔄 Merge %d: %d segments (%.1f MB)", currentMergeCount,
                            merge.segments.size(), sizeMB));
                }

                super.doMerge(mergeSource, merge);

                long mergeTime = System.currentTimeMillis() - startMerge;
                totalMergeTime.addAndGet(mergeTime);

                if (shouldLog) {
                    double throughputMBps = sizeMB / (mergeTime / 1000.0);
                    System.out.println(String.format("✅ Merge %d completed: %,d ms (%.1f MB/s)", currentMergeCount,
                            mergeTime, throughputMBps));
                }
            }
        };
        // Use ALL available threads for merging to maximize speed
        int availableCores = Runtime.getRuntime().availableProcessors();
        mergeScheduler.setMaxMergesAndThreads(availableCores * 2, availableCores);
        indexConfig.setMergeScheduler(mergeScheduler);

        // Lucene will handle thread concurrency internally

        System.out.println("🚀 In-memory config: 512MB RAM buffer, tiered merges; HNSW M=" + config.getHnswMaxConn()
                + ", beamWidth=" + config.getHnswBeamWidth() + "; " + availableCores + " available cores");

        // Monitor memory pressure and GC activity for adaptive indexing
        Runtime runtime = Runtime.getRuntime();
        long maxHeap = runtime.maxMemory();
        long gcThresholdBytes = (long)(maxHeap * 0.7); // Pause at 70% heap usage
        System.out.println(String.format("💾 Memory monitoring: Max heap %,d MB, pause threshold at %,d MB",
                maxHeap / (1024 * 1024), gcThresholdBytes / (1024 * 1024)));

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

        ProgressLogger indexProgress = new ProgressLogger("Indexing " + precision.getName().toUpperCase(),
                numBaseVectors);

        // Parallel indexing with clean work partitioning
        if (numThreads == 1) {
            // Single-threaded indexing
            if (precision == BenchmarkConfig.Precision.F32) {
                float[] vectorBuffer = new float[baseVectors.getCols()];
                for (int i = 0; i < numBaseVectors; i++) {
                    checkMemoryPressure(indexWriter, gcThresholdBytes, i);
                    baseVectors.getVectorAsFloat(i, vectorBuffer);
                    long finalId = (vectorIds != null && i < vectorIds.getNumVectors()) ? vectorIds.getId(i) : i;
                    Document document = new Document();
                    document.add(new NumericDocValuesField("id", finalId));
                    document.add(new KnnFloatVectorField("vector", vectorBuffer));
                    indexWriter.addDocument(document);
                    indexProgress.increment();
                }
            } else if (precision == BenchmarkConfig.Precision.I8) {
                byte[] vectorBuffer = new byte[baseVectors.getCols()];
                for (int i = 0; i < numBaseVectors; i++) {
                    checkMemoryPressure(indexWriter, gcThresholdBytes, i);
                    baseVectors.getVectorAsByte(i, vectorBuffer);
                    long finalId = (vectorIds != null && i < vectorIds.getNumVectors()) ? vectorIds.getId(i) : i;
                    Document document = new Document();
                    document.add(new NumericDocValuesField("id", finalId));
                    document.add(new KnnByteVectorField("vector", vectorBuffer));
                    indexWriter.addDocument(document);
                    indexProgress.increment();
                }
            } else {
                throw new UnsupportedOperationException("Lucene precision " + precision + " not supported yet");
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
                            if (precision == BenchmarkConfig.Precision.F32) {
                                float[] vectorBuffer = new float[baseVectors.getCols()];
                                float[] reusedVector = new float[baseVectors.getCols()];
                                for (int i = startIdx; i < endIdx; i++) {
                                    // Check memory pressure periodically
                                    checkMemoryPressure(indexWriter, gcThresholdBytes, i);

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
                            } else if (precision == BenchmarkConfig.Precision.I8) {
                                byte[] vectorBuffer = new byte[baseVectors.getCols()];
                                byte[] reusedVector = new byte[baseVectors.getCols()];
                                for (int i = startIdx; i < endIdx; i++) {
                                    // Check memory pressure periodically
                                    checkMemoryPressure(indexWriter, gcThresholdBytes, i);

                                    baseVectors.getVectorAsByte(i, reusedVector);
                                    System.arraycopy(reusedVector, 0, vectorBuffer, 0, baseVectors.getCols());
                                    long finalId = (vectorIds != null && i < vectorIds.getNumVectors())
                                            ? vectorIds.getId(i)
                                            : i;
                                    Document document = new Document();
                                    document.add(new NumericDocValuesField("id", finalId));
                                    document.add(new KnnByteVectorField("vector", vectorBuffer));
                                    indexWriter.addDocument(document);
                                    indexProgress.increment();
                                }
                            } else {
                                throw new UnsupportedOperationException(
                                        "Lucene precision " + precision + " not supported yet");
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

        // Commit changes
        indexWriter.commit();

        // Optional: Force merge to target segment count if we have too many segments
        int targetSegments = (int)(estimatedIndexSizeMB / 4096);
        if (targetSegments < 1)
            targetSegments = 1;

        // Check current segment count before deciding to force merge
        DirectoryReader tempReader = DirectoryReader.open(indexWriter);
        int currentSegments = tempReader.leaves().size();
        tempReader.close();

        if (currentSegments > targetSegments * 2) {
            System.out.println(String.format("🔧 Force merging %d segments to %d target segments...", currentSegments,
                    targetSegments));
            long mergeStart = System.currentTimeMillis();
            indexWriter.forceMerge(targetSegments);
            long mergeTime = System.currentTimeMillis() - mergeStart;
            System.out.println(String.format("✅ Force merge completed in %,d ms", mergeTime));
        }

        indexWriter.close();

        long indexingTime = System.currentTimeMillis() - startIndexing;
        long totalMergeTimeValue = totalMergeTime.get();
        long pureIndexingTime = indexingTime - totalMergeTimeValue;

        // Report merge statistics
        if (totalMergeTimeValue > 0) {
            double mergeOverheadPercent = (totalMergeTimeValue * 100.0) / indexingTime;
            double pureIPS = pureIndexingTime > 0 ? (numBaseVectors * 1000.0) / pureIndexingTime : 0.0;
            double totalIPS = indexingTime > 0 ? (numBaseVectors * 1000.0) / indexingTime : 0.0;
            System.out.println(String.format(
                    "⏱️  Merge overhead: %,d ms (%.1f%%) from %d merges - Pure IPS: %.0f, Total IPS: %.0f",
                    totalMergeTimeValue, mergeOverheadPercent, mergeCount.get(), pureIPS, totalIPS));
        }

        long memoryAfter = getMemoryUsage();
        // Use Math.max to prevent negative memory usage due to GC
        long memoryUsageHeap = Math.max(0, memoryAfter - memoryBefore);
        long indexBytes = 0;
        if (directory instanceof DynamicByteBuffersDirectory) {
            indexBytes = ((DynamicByteBuffersDirectory)directory).estimatedSizeInBytes();
        }
        long memoryUsage = Math.max(memoryUsageHeap, indexBytes);

        // Create index reader
        DirectoryReader indexReader = DirectoryReader.open(directory);

        // Report number of segments/leaves for parallelism insight
        int leafCount = indexReader.leaves().size();
        System.out.println(String.format("🪵 Index segments: %d", leafCount));

        // Configure IndexSearcher to use all available cores
        int availableThreads = Runtime.getRuntime().availableProcessors();

        IndexSearcher indexSearcher;
        if (leafCount > 1) {
            java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors
                    .newFixedThreadPool(availableThreads);
            indexSearcher = new IndexSearcher(indexReader, executor);
            System.out.println("🔧 IndexSearcher threads: " + availableThreads + " for " + leafCount + " segments");
        } else {
            indexSearcher = new IndexSearcher(indexReader);
        }

        // Enable query cache for filter-like components (if any)
        indexSearcher.setQueryCache(new LRUQueryCache(4096, 256L * 1024 * 1024)); // 256 MB
        indexSearcher.setQueryCachingPolicy(new UsageTrackingQueryCachingPolicy());

        // Calculate search metrics (includes both search and accuracy calculation)
        System.out.println("🔍 Searching...");
        SearchMetrics searchMetrics = calculateSearchMetrics(indexSearcher, queryVectors, numQueries,
                config.getKValues());

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
                "✅ Lucene HNSW benchmark completed - Indexing: %,dms, Pure Search: %,dms (%,.0f QPS), ID Retrieval: %,dms (%,.0f QPS), Total Search: %,dms (%,.0f QPS)",
                indexingTime, searchMetrics.pureSearchTimeMs, pureSearchQPS, searchMetrics.idRetrievalTimeMs,
                idRetrievalQPS, searchTime, throughputQPS));

        // Per-core accounting, for clarity
        int indexingThreadsUsed = (config.getNumThreads() != -1)
                ? config.getNumThreads()
                : java.util.concurrent.ForkJoinPool.commonPool().getParallelism();
        double ipsTotal = numBaseVectors / (indexingTime / 1000.0);
        double ipsPerCore = indexingThreadsUsed > 0 ? ipsTotal / indexingThreadsUsed : 0.0;
        double pureQpsPerCore = availableThreads > 0 ? pureSearchQPS / availableThreads : 0.0;
        double idQpsPerCore = availableThreads > 0 ? idRetrievalQPS / availableThreads : 0.0;
        double totalQpsPerCore = availableThreads > 0 ? throughputQPS / availableThreads : 0.0;

        System.out
                .println(String.format("   Threads — indexing: %d, search: %d", indexingThreadsUsed, availableThreads));
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
            BinaryVectorLoader.VectorDataset queryVectors, int numQueries, int[] kValues) throws Exception {
        String vectorFieldName = "vector";

        // IDs are already mapped during indexing - no conversion needed

        // Pre-allocate result arrays to avoid allocation overhead during timing
        long[][] allSearchResults = new long[numQueries][];

        // Find maximum K value for single search optimization
        int maxK = Arrays.stream(kValues).max().orElse(100);

        // Submit queries in batches matching config.getBatchSize()
        int batchSize = config.getBatchSize();
        int numBatches = (numQueries + batchSize - 1) / batchSize;
        int availableCores = Runtime.getRuntime().availableProcessors();

        System.out.println(
                String.format("🔍 Processing %d queries in %d batches of %d", numQueries, numBatches, batchSize));

        // Use all available cores for batch processing
        java.util.concurrent.ExecutorService batchExecutor = java.util.concurrent.Executors
                .newFixedThreadPool(availableCores);

        // Phase 1: Pure HNSW search (no ID retrieval)
        ProgressLogger pureSearchProgress = new ProgressLogger("Pure search k=" + maxK, numQueries,
                ProgressLogger.RateUnit.QPS);
        TopDocs[] pureSearchResults = new TopDocs[numQueries];

        long startPureSearch = System.currentTimeMillis();

        for (int batch = 0; batch < numBatches; batch++) {
            int batchStart = batch * batchSize;
            int batchEnd = Math.min(batchStart + batchSize, numQueries);
            int currentBatchSize = batchEnd - batchStart;

            // Submit all queries in this batch concurrently
            List<CompletableFuture<TopDocs>> batchFutures = new ArrayList<>(currentBatchSize);

            for (int i = batchStart; i < batchEnd; i++) {
                final int queryId = i;

                CompletableFuture<TopDocs> future;
                if (precision == BenchmarkConfig.Precision.F32) {
                    float[] queryVector = new float[queryVectors.getCols()];
                    queryVectors.getVectorAsFloat(queryId, queryVector);

                    future = CompletableFuture.supplyAsync(() -> {
                        KnnFloatVectorQuery query = new KnnFloatVectorQuery(vectorFieldName, queryVector, maxK);
                        try {
                            return indexSearcher.search(query, maxK);
                        } catch (IOException e) {
                            throw new RuntimeException("Search failed for query " + queryId, e);
                        }
                    }, batchExecutor);
                } else if (precision == BenchmarkConfig.Precision.I8) {
                    byte[] queryVector = new byte[queryVectors.getCols()];
                    queryVectors.getVectorAsByte(queryId, queryVector);

                    future = CompletableFuture.supplyAsync(() -> {
                        KnnByteVectorQuery query = new KnnByteVectorQuery(vectorFieldName, queryVector, maxK);
                        try {
                            return indexSearcher.search(query, maxK);
                        } catch (IOException e) {
                            throw new RuntimeException("Search failed for query " + queryId, e);
                        }
                    }, batchExecutor);
                } else {
                    throw new UnsupportedOperationException("Lucene precision " + precision + " not supported yet");
                }

                batchFutures.add(future);
            }

            // Wait for this batch to complete before starting next batch
            try {
                CompletableFuture.allOf(batchFutures.toArray(new CompletableFuture[0])).join();

                // Collect results
                for (int i = 0; i < currentBatchSize; i++) {
                    pureSearchResults[batchStart + i] = batchFutures.get(i).get();
                    pureSearchProgress.increment();
                }
            } catch (Exception e) {
                throw new RuntimeException("Batch " + batch + " failed", e);
            }
        }

        long pureSearchTime = System.currentTimeMillis() - startPureSearch;
        pureSearchProgress.complete(numQueries);
        batchExecutor.shutdown();

        // Phase 2: ID retrieval from search results
        ProgressLogger idRetrievalProgress = new ProgressLogger("ID retrieval k=" + maxK, numQueries,
                ProgressLogger.RateUnit.QPS);
        long startIdRetrieval = System.currentTimeMillis();

        // Use same batch processing approach for ID retrieval
        java.util.concurrent.ExecutorService idExecutor = java.util.concurrent.Executors
                .newFixedThreadPool(availableCores);

        for (int batch = 0; batch < numBatches; batch++) {
            int batchStart = batch * batchSize;
            int batchEnd = Math.min(batchStart + batchSize, numQueries);
            int currentBatchSize = batchEnd - batchStart;

            // Submit all ID retrievals in this batch concurrently
            List<CompletableFuture<long[]>> idFutures = new ArrayList<>(currentBatchSize);

            for (int i = batchStart; i < batchEnd; i++) {
                final int queryId = i;

                CompletableFuture<long[]> future = CompletableFuture.supplyAsync(() -> {
                    try {
                        return extractDocumentIds(pureSearchResults[queryId], maxK, indexSearcher);
                    } catch (IOException e) {
                        throw new RuntimeException("ID retrieval failed for query " + queryId, e);
                    }
                }, idExecutor);

                idFutures.add(future);
            }

            // Wait for this batch to complete
            try {
                CompletableFuture.allOf(idFutures.toArray(new CompletableFuture[0])).join();

                // Collect results
                for (int i = 0; i < currentBatchSize; i++) {
                    allSearchResults[batchStart + i] = idFutures.get(i).get();
                    idRetrievalProgress.increment();
                }
            } catch (Exception e) {
                throw new RuntimeException("ID retrieval batch " + batch + " failed", e);
            }
        }

        long idRetrievalTime = System.currentTimeMillis() - startIdRetrieval;
        idRetrievalProgress.complete(numQueries);
        idExecutor.shutdown();

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
            intArray[i] = (int)value;
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
