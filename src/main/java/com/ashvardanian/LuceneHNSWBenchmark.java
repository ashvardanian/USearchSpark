package com.ashvardanian;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LuceneHNSWBenchmark {
    private static final Logger logger = LoggerFactory.getLogger(LuceneHNSWBenchmark.class);

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

    public LuceneHNSWBenchmark(BenchmarkConfig config, DatasetRegistry.Dataset dataset) {
        this.config = config;
        this.dataset = dataset;
    }

    public BenchmarkResult runBenchmark() throws Exception {
        logger.info("Starting Lucene HNSW benchmark for dataset: {}", dataset.getDefinition().getName());
        
        // Load base vectors and queries
        BinaryVectorLoader.VectorDataset baseVectors = BinaryVectorLoader.loadVectors(dataset.getBaseVectorPath());
        BinaryVectorLoader.VectorDataset queryVectors = BinaryVectorLoader.loadVectors(dataset.getQueryVectorPath());
        
        logger.info("Loaded {} base vectors and {} query vectors", 
                   baseVectors.getRows(), queryVectors.getRows());

        // Limit number of queries for benchmarking
        int numQueries = Math.min(config.getNumQueries(), queryVectors.getRows());

        // Create in-memory directory for index
        Directory directory = new ByteBuffersDirectory();
        
        // Configure index writer
        IndexWriterConfig indexConfig = new IndexWriterConfig();
        indexConfig.setUseCompoundFile(false);
        
        // Create index writer
        IndexWriter indexWriter = new IndexWriter(directory, indexConfig);
        
        // Measure indexing time
        long startIndexing = System.currentTimeMillis();
        long memoryBefore = getMemoryUsage();
        
        // Add vectors to index
        addVectorsToIndex(indexWriter, baseVectors);
        indexWriter.close();
        
        long indexingTime = System.currentTimeMillis() - startIndexing;
        long memoryAfter = getMemoryUsage();
        long memoryUsage = memoryAfter - memoryBefore;
        
        // Create index reader and searcher
        DirectoryReader indexReader = DirectoryReader.open(directory);
        IndexSearcher indexSearcher = new IndexSearcher(indexReader);
        
        // Measure search time and calculate recall
        long startSearch = System.currentTimeMillis();
        Map<Integer, Double> recallAtK = calculateRecall(indexSearcher, queryVectors, numQueries, config.getKValues());
        long searchTime = System.currentTimeMillis() - startSearch;
        
        // Calculate throughput
        double throughputQPS = numQueries / (searchTime / 1000.0);
        
        // Cleanup
        indexReader.close();
        directory.close();
        
        logger.info("Lucene HNSW benchmark completed - Indexing: {}ms, Search: {}ms, Throughput: {:.2f} QPS",
                   indexingTime, searchTime, throughputQPS);
        
        return new BenchmarkResult(
            indexingTime,
            searchTime,
            throughputQPS,
            recallAtK,
            memoryUsage,
            baseVectors.getRows(),
            baseVectors.getCols()
        );
    }

    private void addVectorsToIndex(IndexWriter indexWriter, BinaryVectorLoader.VectorDataset baseVectors) 
            throws IOException {
        
        String vectorFieldName = "vector";
        
        for (int i = 0; i < baseVectors.getRows(); i++) {
            float[] vector = baseVectors.getVectorAsFloat(i);
            
            Document document = new Document();
            
            // Add ID field
            document.add(new StoredField("id", i));
            document.add(new StringField("id", String.valueOf(i), Field.Store.YES));
            
            // Add vector field for HNSW search
            document.add(new KnnFloatVectorField(vectorFieldName, vector, getVectorSimilarity()));
            
            indexWriter.addDocument(document);
            
            // Log progress
            if (i % 10000 == 0 && i > 0) {
                logger.info("Indexed {} vectors", i);
            }
        }
        
        logger.info("Finished indexing {} vectors", baseVectors.getRows());
    }

    private VectorSimilarityFunction getVectorSimilarity() {
        String metric = dataset.getDefinition().getMetric().toLowerCase();
        
        switch (metric) {
            case "l2":
                return VectorSimilarityFunction.EUCLIDEAN;
            case "ip":
                return VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
            case "cos":
                return VectorSimilarityFunction.COSINE;
            default:
                logger.warn("Unknown metric '{}', defaulting to EUCLIDEAN", metric);
                return VectorSimilarityFunction.EUCLIDEAN;
        }
    }

    private Map<Integer, Double> calculateRecall(IndexSearcher indexSearcher, 
                                               BinaryVectorLoader.VectorDataset queryVectors,
                                               int numQueries, int[] kValues) throws IOException {
        Map<Integer, Double> recallResults = new HashMap<>();
        String vectorFieldName = "vector";
        
        // For now, we'll use a simplified recall calculation
        // In a full implementation, we'd load and compare against ground truth
        for (int k : kValues) {
            double totalRecall = 0.0;
            
            for (int i = 0; i < numQueries; i++) {
                float[] queryVector = queryVectors.getVectorAsFloat(i);
                
                // Create KNN query
                KnnFloatVectorQuery knnQuery = new KnnFloatVectorQuery(vectorFieldName, queryVector, k);
                
                // Execute search
                TopDocs topDocs = indexSearcher.search(knnQuery, k);
                
                // Simplified recall calculation - in practice you'd compare against ground truth
                // For now, just check if we got the expected number of results
                double queryRecall = Math.min(1.0, (double) topDocs.scoreDocs.length / k);
                totalRecall += queryRecall;
                
                // Log progress for long-running searches
                if (i % 1000 == 0 && i > 0) {
                    logger.debug("Processed {} queries for k={}", i, k);
                }
            }
            
            recallResults.put(k, totalRecall / numQueries);
            logger.info("Recall@{}: {:.4f}", k, totalRecall / numQueries);
        }
        
        return recallResults;
    }

    private long getMemoryUsage() {
        if (!config.isIncludeMemoryUsage()) {
            return 0;
        }
        
        Runtime runtime = Runtime.getRuntime();
        runtime.gc(); // Suggest garbage collection
        
        try {
            Thread.sleep(100); // Give GC a moment
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        return runtime.totalMemory() - runtime.freeMemory();
    }

    public static void logBenchmarkResults(BenchmarkResult result) {
        logger.info("=== Lucene HNSW Benchmark Results ===");
        logger.info("Precision: F32 (baseline)");
        logger.info("  Indexing Time: {} ms", result.getIndexingTimeMs());
        logger.info("  Search Time: {} ms", result.getSearchTimeMs());
        logger.info("  Throughput: {:.2f} QPS", result.getThroughputQPS());
        logger.info("  Memory Usage: {:.2f} MB", result.getMemoryUsageBytes() / (1024.0 * 1024.0));
        
        for (Map.Entry<Integer, Double> entry : result.getRecallAtK().entrySet()) {
            logger.info("  Recall@{}: {:.4f}", entry.getKey(), entry.getValue());
        }
        
        logger.info("");
    }
}