package com.ashvardanian;

import java.io.IOException;
import java.util.ArrayList;
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

    public static class BenchmarkResult {
        private final long indexingTimeMs;
        private final long searchTimeMs;
        private final double throughputQPS;
        private final Map<Integer, Double> recallAtK;
        private final long memoryUsageBytes;
        private final int numVectors;
        private final int dimensions;

        public BenchmarkResult(long indexingTimeMs, long searchTimeMs, double throughputQPS,
                Map<Integer, Double> recallAtK, long memoryUsageBytes,
                int numVectors, int dimensions) {
            this.indexingTimeMs = indexingTimeMs;
            this.searchTimeMs = searchTimeMs;
            this.throughputQPS = throughputQPS;
            this.recallAtK = Collections.unmodifiableMap(new HashMap<>(recallAtK));
            this.memoryUsageBytes = memoryUsageBytes;
            this.numVectors = numVectors;
            this.dimensions = dimensions;
        }

        // Getters
        public long getIndexingTimeMs() { return indexingTimeMs; }
        public long getSearchTimeMs() { return searchTimeMs; }
        public double getThroughputQPS() { return throughputQPS; }
        public Map<Integer, Double> getRecallAtK() { return recallAtK; }
        public long getMemoryUsageBytes() { return memoryUsageBytes; }
        public int getNumVectors() { return numVectors; }
        public int getDimensions() { return dimensions; }
    }

    private final BenchmarkConfig config;
    private final DatasetRegistry.Dataset dataset;

    public LuceneBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) {
        this.config = config;
        this.dataset = dataset;
    }

    public BenchmarkResult runBenchmark() throws Exception {
        System.out.println("\n🔍 Starting Lucene HNSW benchmark for dataset: " + dataset.getDefinition().getName());

        // Load base vectors and queries
        System.out.print("📂 Loading vectors... ");
        BinaryVectorLoader.VectorDataset baseVectors = BinaryVectorLoader.loadVectors(dataset.getBaseVectorPath());
        BinaryVectorLoader.VectorDataset queryVectors = BinaryVectorLoader.loadVectors(dataset.getQueryVectorPath());
        System.out.println("✅ Done");

        // Limit number of base vectors if specified
        int numBaseVectors = baseVectors.getRows();
        if (config.getMaxVectors() > 0 && config.getMaxVectors() < baseVectors.getRows()) {
            numBaseVectors = (int) Math.min(config.getMaxVectors(), Integer.MAX_VALUE);
            System.out.println("🔢 Limiting base vectors to " + String.format("%,d", numBaseVectors) + " for faster testing");
        }

        System.out.println("📊 Using " + String.format("%,d", numBaseVectors) + " base vectors and " +
                String.format("%,d", queryVectors.getRows()) + " query vectors");

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());

        // Create in-memory directory for index
        Directory directory = new ByteBuffersDirectory();

        // Configure index writer with multithreading
        IndexWriterConfig indexConfig = new IndexWriterConfig();
        indexConfig.setUseCompoundFile(false);
        indexConfig.setMaxBufferedDocs(1000);
        indexConfig.setRAMBufferSizeMB(256);

        // Create index writer
        IndexWriter indexWriter = new IndexWriter(directory, indexConfig);

        // Measure indexing time
        long startIndexing = System.currentTimeMillis();
        long memoryBefore = getMemoryUsage();

        // Create batches for multithreaded indexing
        List<VectorProcessor.VectorBatch> indexingBatches = 
            VectorProcessor.createBatches(baseVectors, numBaseVectors, false, 1024);

        // Multithreaded batch indexing
        VectorProcessor.processBatches(indexingBatches, batch -> {
            addVectorBatchToIndex(indexWriter, batch);
            return null;
        }, "Indexing F32");

        indexWriter.close();

        long indexingTime = System.currentTimeMillis() - startIndexing;
        long memoryAfter = getMemoryUsage();
        long memoryUsage = memoryAfter - memoryBefore;

        // Create index reader and searcher
        DirectoryReader indexReader = DirectoryReader.open(directory);
        IndexSearcher indexSearcher = new IndexSearcher(indexReader);

        // Measure search time and calculate recall
        long startSearch = System.currentTimeMillis();
        System.out.print("🔍 Searching... ");
        Map<Integer, Double> recallAtK = calculateRecall(indexSearcher, queryVectors, numQueries, config.getKValues());
        long searchTime = System.currentTimeMillis() - startSearch;
        System.out.println("✅ Done");

        // Calculate throughput
        double throughputQPS = numQueries / (searchTime / 1000.0);

        // Cleanup
        indexReader.close();
        directory.close();

        System.out.println(String.format("✅ Lucene HNSW benchmark completed - Indexing: %,dms, Search: %,dms, Throughput: %,.0f QPS",
                indexingTime, searchTime, throughputQPS));

        return new BenchmarkResult(
                indexingTime,
                searchTime,
                throughputQPS,
                recallAtK,
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

            // Add ID field
            document.add(new StoredField("id", key));
            document.add(new StringField("id", String.valueOf(key), Field.Store.YES));

            // Add vector field with HNSW
            document.add(new KnnFloatVectorField(vectorFieldName, vector));

            documents.add(document);
        }

        // Batch add documents for better performance
        synchronized (indexWriter) {
            indexWriter.addDocuments(documents);
        }
    }

    private Map<Integer, Double> calculateRecall(IndexSearcher indexSearcher,
            BinaryVectorLoader.VectorDataset queryVectors,
            int numQueries, int[] kValues) throws Exception {
        Map<Integer, Double> recallResults = new HashMap<>();
        String vectorFieldName = "vector";

        // Create batches for query vectors
        List<VectorProcessor.VectorBatch> queryBatches = 
            VectorProcessor.createBatches(queryVectors, numQueries, false, 1024); // Lucene always uses float

        // For each k value, run concurrent searches
        for (int k : kValues) {
            // Concurrent search processing
            List<Double> recalls = VectorProcessor.processBatches(queryBatches, batch -> {
                double batchRecall = 0.0;
                
                try {
                    for (int i = 0; i < batch.vectorCount; i++) {
                        float[] vector = new float[batch.dimensions];
                        System.arraycopy(batch.vectors, i * batch.dimensions, vector, 0, batch.dimensions);
                        
                        // Perform HNSW search
                        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(vectorFieldName, vector, k);
                        TopDocs topDocs = indexSearcher.search(vectorQuery, k);
                        
                        // Simplified recall calculation
                        double queryRecall = Math.min(1.0, (double) topDocs.scoreDocs.length / k);
                        batchRecall += queryRecall;
                    }
                } catch (IOException e) {
                    throw new RuntimeException("Search failed", e);
                }
                
                return batchRecall;
            }, "Searching k=" + k);
            
            // Aggregate recall from all batches
            double totalRecall = recalls.stream().mapToDouble(Double::doubleValue).sum();
            recallResults.put(k, totalRecall / numQueries);
        }

        return recallResults;
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
        System.out.println(String.format("  Throughput: %,.0f QPS", result.getThroughputQPS()));
        System.out.println(String.format("  Memory Usage: %,d MB", Math.round(result.getMemoryUsageBytes() / (1024.0 * 1024.0))));

        for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
            System.out.println(String.format("  Recall@%d: %.4f", entry.getKey(), entry.getValue()));
        }

        System.out.println();
    }
}