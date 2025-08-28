package com.ashvardanian;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Multithreaded batch processor for vector operations */
public class VectorProcessor {
    private static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();

    public static class VectorBatch {
        public final long[] keys;
        public final float[] vectors; // For F32, F16, BF16
        public final byte[] byteVectors; // For I8
        public final boolean isByteData;
        public final int vectorCount;
        public final int dimensions;
        public final int startIndex; // Starting index in original dataset

        public VectorBatch(
                long[] keys, float[] vectors, int vectorCount, int dimensions, int startIndex) {
            this.keys = keys;
            this.vectors = vectors;
            this.byteVectors = null;
            this.isByteData = false;
            this.vectorCount = vectorCount;
            this.dimensions = dimensions;
            this.startIndex = startIndex;
        }

        public VectorBatch(
                long[] keys, byte[] byteVectors, int vectorCount, int dimensions, int startIndex) {
            this.keys = keys;
            this.vectors = null;
            this.byteVectors = byteVectors;
            this.isByteData = true;
            this.vectorCount = vectorCount;
            this.dimensions = dimensions;
            this.startIndex = startIndex;
        }
    }

    public static class ProgressTracker {
        private final AtomicInteger processedBatches = new AtomicInteger(0);
        private final AtomicLong processedVectors = new AtomicLong(0);
        private final AtomicLong startTime = new AtomicLong(System.currentTimeMillis());
        private final int totalBatches;
        private final long totalVectors;
        private final String operationName;

        public ProgressTracker(int totalBatches, long totalVectors, String operationName) {
            this.totalBatches = totalBatches;
            this.totalVectors = totalVectors;
            this.operationName = operationName;
        }

        public void onBatchCompleted(int batchSize) {
            int batches = processedBatches.incrementAndGet();
            long vectors = processedVectors.addAndGet(batchSize);

            if (batches % 10 == 0 || batches == totalBatches) {
                long elapsedMs = System.currentTimeMillis() - startTime.get();
                double ips = elapsedMs > 0 ? (vectors * 1000.0) / elapsedMs : 0;
                double progress = (double) batches / totalBatches * 100;

                System.out.printf(
                        "\r🔄 %s: %.1f%% (%d/%d batches, %,d vectors, %.0f IPS)",
                        operationName, progress, batches, totalBatches, vectors, ips);

                if (batches == totalBatches) {
                    System.out.println();
                }
            }
        }

        public void completed() {
            long elapsedMs = System.currentTimeMillis() - startTime.get();
            double ips = elapsedMs > 0 ? (totalVectors * 1000.0) / elapsedMs : 0;
            System.out.printf(
                    "✅ %s completed: %,d vectors in %,dms (%.0f IPS)%n",
                    operationName, totalVectors, elapsedMs, ips);
        }
    }

    /** Create batches from vector dataset */
    public static List<VectorBatch> createBatches(
            BinaryVectorLoader.VectorDataset dataset,
            int maxVectors,
            boolean useByteData,
            int batchSize) {
        int numVectors =
                maxVectors > 0 ? Math.min(maxVectors, dataset.getRows()) : dataset.getRows();
        int dimensions = dataset.getCols();
        List<VectorBatch> batches = new ArrayList<>();

        for (int start = 0; start < numVectors; start += batchSize) {
            int end = Math.min(start + batchSize, numVectors);
            int currentBatchSize = end - start;

            long[] keys = new long[currentBatchSize];
            for (int i = 0; i < currentBatchSize; i++) {
                keys[i] = start + i;
            }

            if (useByteData) {
                byte[] batchVectors = new byte[currentBatchSize * dimensions];
                for (int i = 0; i < currentBatchSize; i++) {
                    float[] vector = dataset.getVectorAsFloat(start + i);
                    for (int j = 0; j < dimensions; j++) {
                        // Convert float to byte for I8 quantization
                        batchVectors[i * dimensions + j] = (byte) Math.round(vector[j] * 127.0f);
                    }
                }
                batches.add(
                        new VectorBatch(keys, batchVectors, currentBatchSize, dimensions, start));
            } else {
                float[] batchVectors = new float[currentBatchSize * dimensions];
                for (int i = 0; i < currentBatchSize; i++) {
                    float[] vector = dataset.getVectorAsFloat(start + i);
                    System.arraycopy(vector, 0, batchVectors, i * dimensions, dimensions);
                }
                batches.add(
                        new VectorBatch(keys, batchVectors, currentBatchSize, dimensions, start));
            }
        }

        return batches;
    }

    /** Process batches using multiple threads with a custom batch processor */
    public static <T> List<T> processBatches(
            List<VectorBatch> batches,
            BatchProcessor<T> processor,
            String operationName,
            int threadCount)
            throws Exception {
        if (batches.isEmpty()) {
            return new ArrayList<>();
        }

        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        List<Future<T>> futures = new ArrayList<>();

        long totalVectors = batches.stream().mapToLong(b -> b.vectorCount).sum();
        ProgressTracker tracker = new ProgressTracker(batches.size(), totalVectors, operationName);

        try {
            // Submit all batch processing tasks
            for (VectorBatch batch : batches) {
                futures.add(
                        executor.submit(
                                () -> {
                                    try {
                                        T result = processor.processBatch(batch);
                                        tracker.onBatchCompleted(batch.vectorCount);
                                        return result;
                                    } catch (Exception e) {
                                        throw new RuntimeException("Failed to process batch", e);
                                    }
                                }));
            }

            // Collect results
            List<T> results = new ArrayList<>();
            for (Future<T> future : futures) {
                results.add(future.get());
            }

            tracker.completed();
            return results;

        } finally {
            executor.shutdown();
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        }
    }

    /** Process batches using multiple threads with default thread count */
    public static <T> List<T> processBatches(
            List<VectorBatch> batches, BatchProcessor<T> processor, String operationName)
            throws Exception {
        return processBatches(batches, processor, operationName, DEFAULT_THREAD_COUNT);
    }

    @FunctionalInterface
    public interface BatchProcessor<T> {
        T processBatch(VectorBatch batch) throws Exception;
    }
}
