package com.ashvardanian;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BinaryVectorLoader {
    private static final Logger logger = LoggerFactory.getLogger(BinaryVectorLoader.class);

    public enum VectorType {
        FLOAT32(".fbin", 4, Float.BYTES), FLOAT64(".dbin", 8, Double.BYTES), FLOAT16(".hbin", 2, 2), // Half precision
        INT32(".ibin", 4, Integer.BYTES), UINT8(".bbin", 1, Byte.BYTES), INT8(".i8bin", 1, Byte.BYTES), // Signed 8-bit
                                                                                                        // integers
                                                                                                        // (SPACEV
                                                                                                        // format)
        FLOAT32_BIN(".f32bin", 4, Float.BYTES), // Alternative F32 format (SpaceV distances)
        INT32_BIN(".i32bin", 4, Integer.BYTES), // Alternative I32 format (SpaceV ground truth)
        UINT8_BIN(".u8bin", 1, Byte.BYTES); // Alternative uint8 format

        private final String extension;
        private final int byteSize;
        private final int javaTypeSize;

        VectorType(String extension, int byteSize, int javaTypeSize) {
            this.extension = extension;
            this.byteSize = byteSize;
            this.javaTypeSize = javaTypeSize;
        }

        public String getExtension() {
            return extension;
        }

        public int getByteSize() {
            return byteSize;
        }

        public int getJavaTypeSize() {
            return javaTypeSize;
        }

        public static VectorType fromPath(String path) {
            String lowerPath = path.toLowerCase();
            for (VectorType type : values()) {
                if (lowerPath.endsWith(type.extension)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unknown vector file type: " + path);
        }
    }

    public static class VectorDataset {
        private final int rows;
        private final int cols;
        private final VectorType type;
        private final ByteBuffer[] segments; // Multiple segments for large files
        private final int startRow;
        private final long segmentSize;

        public VectorDataset(int rows, int cols, VectorType type, ByteBuffer[] segments, int startRow,
                long segmentSize) {
            this.rows = rows;
            this.cols = cols;
            this.type = type;
            this.segments = segments;
            this.startRow = startRow;
            this.segmentSize = segmentSize;
        }

        // Backward compatibility constructor for single segment
        public VectorDataset(int rows, int cols, VectorType type, ByteBuffer data, int startRow) {
            this(rows, cols, type, new ByteBuffer[]{data}, startRow, data.remaining());
        }

        public int getRows() {
            return rows;
        }

        public int getCols() {
            return cols;
        }

        public VectorType getType() {
            return type;
        }

        public ByteBuffer getData() {
            // Return first segment for backward compatibility
            return segments.length > 0 ? segments[0] : null;
        }

        public float[] getVectorAsFloat(int index) {
            if (index >= rows) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + rows);
            }

            float[] result = new float[cols];
            getVectorAsFloat(index, result);
            return result;
        }

        public void getVectorAsFloat(int index, float[] result) {
            if (index >= rows) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + rows);
            }
            if (result.length < cols) {
                throw new IllegalArgumentException("Buffer too small: " + result.length + " < " + cols);
            }

            long offset = (long) index * cols * type.getByteSize();

            switch (type) {
                case FLOAT32 :
                    for (int i = 0; i < cols; i++) {
                        result[i] = getFloat(offset + i * Float.BYTES);
                    }
                    break;
                case FLOAT64 :
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) getDouble(offset + i * Double.BYTES);
                    }
                    break;
                case INT32 :
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) getInt(offset + i * Integer.BYTES);
                    }
                    break;
                case INT8 :
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) getByte(offset + i); // Signed byte
                    }
                    break;
                case UINT8 :
                case UINT8_BIN :
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) (getByte(offset + i) & 0xFF); // Unsigned byte
                    }
                    break;
                case FLOAT16 :
                    for (int i = 0; i < cols; i++) {
                        short halfFloat = getShort(offset + i * 2);
                        result[i] = halfFloatToFloat(halfFloat);
                    }
                    break;
                default :
                    throw new UnsupportedOperationException("Unsupported vector type: " + type);
            }
        }

        // Helper methods to handle long offsets across multiple memory-mapped segments
        // This approach handles files larger than 2GB by splitting them into multiple
        // segments
        // Reference:
        // https://blog.vanillajava.blog/2011/12/using-memory-mapped-file-for-huge.html
        private float getFloat(long offset) {
            int segmentIndex = (int) (offset / segmentSize);
            int segmentOffset = (int) (offset % segmentSize);

            if (segmentIndex >= segments.length) {
                throw new IllegalArgumentException("Offset beyond mapped segments: " + offset);
            }
            if (segmentOffset > segments[segmentIndex].limit() - Float.BYTES) {
                throw new IllegalArgumentException("Float read would exceed segment boundary");
            }

            return segments[segmentIndex].getFloat(segmentOffset);
        }

        private double getDouble(long offset) {
            int segmentIndex = (int) (offset / segmentSize);
            int segmentOffset = (int) (offset % segmentSize);

            if (segmentIndex >= segments.length) {
                throw new IllegalArgumentException("Offset beyond mapped segments: " + offset);
            }
            if (segmentOffset > segments[segmentIndex].limit() - Double.BYTES) {
                throw new IllegalArgumentException("Double read would exceed segment boundary");
            }

            return segments[segmentIndex].getDouble(segmentOffset);
        }

        private int getInt(long offset) {
            int segmentIndex = (int) (offset / segmentSize);
            int segmentOffset = (int) (offset % segmentSize);

            if (segmentIndex >= segments.length) {
                throw new IllegalArgumentException("Offset beyond mapped segments: " + offset);
            }
            if (segmentOffset > segments[segmentIndex].limit() - Integer.BYTES) {
                throw new IllegalArgumentException("Int read would exceed segment boundary");
            }

            return segments[segmentIndex].getInt(segmentOffset);
        }

        private byte getByte(long offset) {
            int segmentIndex = (int) (offset / segmentSize);
            int segmentOffset = (int) (offset % segmentSize);

            if (segmentIndex >= segments.length) {
                throw new IllegalArgumentException("Offset beyond mapped segments: " + offset);
            }
            if (segmentOffset >= segments[segmentIndex].limit()) {
                throw new IllegalArgumentException("Byte read would exceed segment boundary");
            }

            return segments[segmentIndex].get(segmentOffset);
        }

        private short getShort(long offset) {
            int segmentIndex = (int) (offset / segmentSize);
            int segmentOffset = (int) (offset % segmentSize);

            if (segmentIndex >= segments.length) {
                throw new IllegalArgumentException("Offset beyond mapped segments: " + offset);
            }
            if (segmentOffset > segments[segmentIndex].limit() - Short.BYTES) {
                throw new IllegalArgumentException("Short read would exceed segment boundary");
            }

            return segments[segmentIndex].getShort(segmentOffset);
        }

        public byte[] getVectorAsByte(int index) {
            if (index >= rows) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + rows);
            }

            byte[] result = new byte[cols];
            getVectorAsByte(index, result);
            return result;
        }

        public void getVectorAsByte(int index, byte[] result) {
            if (index >= rows) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + rows);
            }
            if (result.length < cols) {
                throw new IllegalArgumentException("Buffer too small: " + result.length + " < " + cols);
            }

            long offset = (long) index * cols * type.getByteSize();

            switch (type) {
                case INT8 :
                    for (int i = 0; i < cols; i++) {
                        result[i] = getByte(offset + i);
                    }
                    break;
                case FLOAT32 :
                    // Convert float to byte for I8 quantization
                    for (int i = 0; i < cols; i++) {
                        float value = getFloat(offset + i * Float.BYTES);
                        result[i] = (byte) Math.round(value * 127.0f);
                    }
                    break;
                default :
                    // Convert other types to byte via float - use reusable buffer
                    float[] tempFloatBuffer = new float[cols];
                    getVectorAsFloat(index, tempFloatBuffer);
                    for (int i = 0; i < cols; i++) {
                        result[i] = (byte) Math.round(tempFloatBuffer[i] * 127.0f);
                    }
                    break;
            }
        }

        public boolean isI8Data() {
            return type == VectorType.INT8;
        }

        public float[][] getAllVectors() {
            float[][] result = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                result[i] = getVectorAsFloat(i);
            }
            return result;
        }
    }

    public static VectorDataset loadVectors(String filePath) throws IOException {
        return loadVectors(filePath, 0, -1);
    }

    public static VectorDataset loadVectors(String filePath, int startRow, int maxRows) throws IOException {
        Path path = Paths.get(filePath);
        VectorType type = VectorType.fromPath(filePath);

        logger.debug("Loading vector file: {} (type: {})", filePath, type);

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // Read header (8 bytes: rows and columns as 32-bit integers)
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();

            int totalRows = header.getInt();
            int cols = header.getInt();

            logger.debug("Dataset dimensions: {} x {} ({})", totalRows, cols, type);

            int actualRows = maxRows > 0 ? Math.min(maxRows, totalRows - startRow) : totalRows - startRow;
            if (startRow >= totalRows) {
                throw new IndexOutOfBoundsException("Start row " + startRow + " >= total rows " + totalRows);
            }

            // Calculate data size and offset
            long vectorDataSize = (long) totalRows * cols * type.getByteSize();
            long startOffset = 8 + (long) startRow * cols * type.getByteSize();
            long readSize = (long) actualRows * cols * type.getByteSize();

            // Handle large files using multiple memory-mapped segments
            // Java's FileChannel.map() has a 2GB (Integer.MAX_VALUE) limit per mapping
            // For larger files, we create multiple segments as described in:
            // https://blog.vanillajava.blog/2011/12/using-memory-mapped-file-for-huge.html

            // JVM ByteBuffer limitation - max ~2GB per segment
            final long maxSegmentSize = Integer.MAX_VALUE - 1024;
            ByteBuffer[] segments;
            long segmentSize;

            if (readSize > maxSegmentSize) {
                // Calculate number of segments needed
                int numSegments = (int) Math.ceil((double) readSize / maxSegmentSize);
                segments = new ByteBuffer[numSegments];
                segmentSize = maxSegmentSize;

                logger.info("Large file detected ({} GB). Creating {} memory-mapped segments of {} MB each",
                        String.format("%.1f", readSize / (1024 * 1024 * 1024.0)), numSegments,
                        String.format("%.0f", maxSegmentSize / (1024 * 1024.0)));

                // Create multiple memory-mapped segments
                for (int i = 0; i < numSegments; i++) {
                    long segmentOffset = startOffset + (i * maxSegmentSize);
                    long currentSegmentSize = Math.min(maxSegmentSize, readSize - (i * maxSegmentSize));

                    MappedByteBuffer segment = channel.map(FileChannel.MapMode.READ_ONLY, segmentOffset,
                            currentSegmentSize);
                    segment.order(ByteOrder.LITTLE_ENDIAN);
                    segments[i] = segment;
                }
            } else {
                // Single segment for files under 2GB
                segments = new ByteBuffer[1];
                segmentSize = readSize;

                MappedByteBuffer segment = channel.map(FileChannel.MapMode.READ_ONLY, startOffset, readSize);
                segment.order(ByteOrder.LITTLE_ENDIAN);
                segments[0] = segment;
            }

            logger.debug("Loaded {} vectors starting from row {} using {} segment(s)", actualRows, startRow,
                    segments.length);
            return new VectorDataset(actualRows, cols, type, segments, startRow, segmentSize);
        }
    }

    // Convert IEEE 754 half precision to single precision
    private static float halfFloatToFloat(short halfFloat) {
        int sign = (halfFloat & 0x8000) << 16;
        int exponent = (halfFloat & 0x7C00) >> 10;
        int mantissa = halfFloat & 0x03FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(sign); // Zero
            } else {
                // Denormalized number
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                exponent++;
                mantissa &= ~0x400;
            }
        } else if (exponent == 31) {
            // Infinity or NaN
            return Float.intBitsToFloat(sign | 0x7F800000 | (mantissa << 13));
        }

        exponent += (127 - 15);
        return Float.intBitsToFloat(sign | (exponent << 23) | (mantissa << 13));
    }

    public static class DatasetInfo {
        private final String path;
        private final int rows;
        private final int cols;
        private final VectorType type;

        public DatasetInfo(String path, int rows, int cols, VectorType type) {
            this.path = path;
            this.rows = rows;
            this.cols = cols;
            this.type = type;
        }

        public String getPath() {
            return path;
        }

        public int getRows() {
            return rows;
        }

        public int getCols() {
            return cols;
        }

        public VectorType getType() {
            return type;
        }
    }

    public static DatasetInfo getDatasetInfo(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        VectorType type = VectorType.fromPath(filePath);

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();

            int rows = header.getInt();
            int cols = header.getInt();

            return new DatasetInfo(filePath, rows, cols, type);
        }
    }

    public static class GroundTruth {
        private final int numQueries;
        private final int k;
        private final VectorType type;
        private final ByteBuffer data;

        public GroundTruth(int numQueries, int k, VectorType type, ByteBuffer data) {
            this.numQueries = numQueries;
            this.k = k;
            this.type = type;
            this.data = data;
        }

        public int getNumQueries() {
            return numQueries;
        }

        public int getK() {
            return k;
        }

        public VectorType getType() {
            return type;
        }

        public int[] getNeighbors(int queryIndex) {
            if (queryIndex >= numQueries) {
                throw new IndexOutOfBoundsException("Query index " + queryIndex + " >= " + numQueries);
            }

            int[] neighbors = new int[k];
            int offset = queryIndex * k * type.getByteSize();

            if (type == VectorType.INT32 || type == VectorType.INT32_BIN) {
                for (int i = 0; i < k; i++) {
                    neighbors[i] = data.getInt(offset + i * Integer.BYTES);
                }
            } else {
                throw new UnsupportedOperationException("Ground truth type " + type + " not supported");
            }

            return neighbors;
        }

        public float[] getDistances(int queryIndex) {
            if (type != VectorType.FLOAT32 && type != VectorType.FLOAT32_BIN) {
                throw new UnsupportedOperationException("Distance extraction only supported for FLOAT32 ground truth");
            }

            float[] distances = new float[k];
            int offset = queryIndex * k * type.getByteSize();

            for (int i = 0; i < k; i++) {
                distances[i] = data.getFloat(offset + i * Float.BYTES);
            }

            return distances;
        }
    }

    public static class VectorIds {
        private final int numVectors;
        private final ByteBuffer data;

        public VectorIds(int numVectors, ByteBuffer data) {
            this.numVectors = numVectors;
            this.data = data;
        }

        public int getNumVectors() {
            return numVectors;
        }

        public int getId(int index) {
            if (index >= numVectors) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + numVectors);
            }
            return data.getInt(index * Integer.BYTES);
        }

        public int[] getIds() {
            int[] ids = new int[numVectors];
            for (int i = 0; i < numVectors; i++) {
                ids[i] = data.getInt(i * Integer.BYTES);
            }
            return ids;
        }
    }

    public static GroundTruth loadGroundTruth(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        VectorType type = VectorType.fromPath(filePath);

        logger.debug("Loading ground truth file: {} (type: {})", filePath, type);

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();

            int numQueries = header.getInt();
            int k = header.getInt();

            logger.debug("Ground truth dimensions: {} queries x {} neighbors ({})", numQueries, k, type);

            long dataSize = (long) numQueries * k * type.getByteSize();
            ByteBuffer data = ByteBuffer.allocate((int) dataSize).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(data);
            data.flip();

            return new GroundTruth(numQueries, k, type, data);
        }
    }

    public static VectorIds loadVectorIds(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        VectorType type = VectorType.fromPath(filePath);

        if (type != VectorType.INT32 && type != VectorType.INT32_BIN) {
            throw new IllegalArgumentException("Vector IDs must be INT32 or INT32_BIN format, got: " + type);
        }

        logger.debug("Loading vector IDs file: {}", filePath);

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();

            int numVectors = header.getInt();
            int cols = header.getInt();

            if (cols != 1) {
                throw new IllegalArgumentException("Expected 1 column for vector IDs, got: " + cols);
            }

            logger.info("Vector IDs: {} entries", numVectors);

            long dataSize = (long) numVectors * Integer.BYTES;
            ByteBuffer data = ByteBuffer.allocate((int) dataSize).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(data);
            data.flip();

            return new VectorIds(numVectors, data);
        }
    }

    /**
     * Calculate Recall@k following true USearch methodology. Measures whether the #1 ground truth neighbor appears
     * anywhere in the top-k results. Returns 1.0 if found, 0.0 if not found (binary per query, averaged across
     * queries).
     */
    public static double calculateRecallAtK(GroundTruth groundTruth, int queryIndex, int[] searchResults, int k) {
        if (searchResults.length == 0 || k == 0) {
            return 0.0;
        }

        int[] trueNeighbors = groundTruth.getNeighbors(queryIndex);
        if (trueNeighbors.length == 0) {
            return 0.0;
        }

        // USearch recall@k: Does the #1 ground truth appear in top-k results?
        int topGroundTruth = trueNeighbors[0]; // Most relevant ground truth item
        int searchK = Math.min(k, searchResults.length);

        for (int i = 0; i < searchK; i++) {
            if (searchResults[i] == topGroundTruth) {
                return 1.0; // Found the top ground truth item
            }
        }

        return 0.0; // Top ground truth item not found in top-k results
    }

    /**
     * Calculate NDCG@k (Normalized Discounted Cumulative Gain) following USearch methodology. Only considers top-k
     * ground truth as relevant items (not exhaustive recall). NDCG considers the position of correct results - earlier
     * positions get higher scores.
     */
    public static double calculateNDCGAtK(GroundTruth groundTruth, int queryIndex, int[] searchResults, int k) {
        if (searchResults.length == 0 || k == 0) {
            return 0.0;
        }

        int[] trueNeighbors = groundTruth.getNeighbors(queryIndex);
        if (trueNeighbors.length == 0) {
            return 0.0;
        }

        // Create relevance map from ONLY top-k ground truth neighbors (USearch philosophy)
        int groundTruthK = Math.min(k, trueNeighbors.length);
        java.util.Set<Integer> relevantSet = new java.util.HashSet<>();
        for (int i = 0; i < groundTruthK; i++) {
            relevantSet.add(trueNeighbors[i]);
        }

        // Calculate DCG (Discounted Cumulative Gain) for search results
        double dcg = 0.0;
        int searchK = Math.min(k, searchResults.length);
        for (int i = 0; i < searchK; i++) {
            double relevance = relevantSet.contains(searchResults[i]) ? 1.0 : 0.0;
            // DCG formula: relevance / log2(position + 1), where position is 1-indexed
            dcg += relevance / (Math.log(i + 2) / Math.log(2));
        }

        // Calculate IDCG (Ideal DCG) - perfect ranking of top-k ground truth items
        // USearch philosophy: only consider intersection of k and available ground truth
        double idcg = 0.0;
        int maxPossibleRelevant = Math.min(k, groundTruthK); // At most k relevant items
        for (int i = 0; i < maxPossibleRelevant; i++) {
            // In ideal ranking, all top-k ground truth items get relevance = 1.0
            idcg += 1.0 / (Math.log(i + 2) / Math.log(2));
        }

        // NDCG = DCG / IDCG
        return idcg > 0 ? dcg / idcg : 0.0;
    }
}
