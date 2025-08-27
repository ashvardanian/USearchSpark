package com.ashvardanian;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BinaryVectorLoader {
    private static final Logger logger = LoggerFactory.getLogger(BinaryVectorLoader.class);

    public enum VectorType {
        FLOAT32(".fbin", 4, Float.BYTES),
        FLOAT64(".dbin", 8, Double.BYTES),
        FLOAT16(".hbin", 2, 2), // Half precision
        INT32(".ibin", 4, Integer.BYTES),
        INT8(".i8bin", 1, Byte.BYTES), // Signed 8-bit integers (SPACEV format)
        UINT8(".bbin", 1, Byte.BYTES),
        UINT8_BIN(".u8bin", 1, Byte.BYTES); // Alternative uint8 format

        private final String extension;
        private final int byteSize;
        private final int javaTypeSize;

        VectorType(String extension, int byteSize, int javaTypeSize) {
            this.extension = extension;
            this.byteSize = byteSize;
            this.javaTypeSize = javaTypeSize;
        }

        public String getExtension() { return extension; }
        public int getByteSize() { return byteSize; }
        public int getJavaTypeSize() { return javaTypeSize; }

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
        private final ByteBuffer data;

        public VectorDataset(int rows, int cols, VectorType type, ByteBuffer data) {
            this.rows = rows;
            this.cols = cols;
            this.type = type;
            this.data = data;
        }

        public int getRows() { return rows; }
        public int getCols() { return cols; }
        public VectorType getType() { return type; }
        public ByteBuffer getData() { return data; }

        public float[] getVectorAsFloat(int index) {
            if (index >= rows) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + rows);
            }

            float[] result = new float[cols];
            int offset = index * cols * type.getByteSize();
            
            switch (type) {
                case FLOAT32:
                    for (int i = 0; i < cols; i++) {
                        result[i] = data.getFloat(offset + i * Float.BYTES);
                    }
                    break;
                case FLOAT64:
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) data.getDouble(offset + i * Double.BYTES);
                    }
                    break;
                case INT32:
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) data.getInt(offset + i * Integer.BYTES);
                    }
                    break;
                case INT8:
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) data.get(offset + i); // Signed byte
                    }
                    break;
                case UINT8:
                case UINT8_BIN:
                    for (int i = 0; i < cols; i++) {
                        result[i] = (float) (data.get(offset + i) & 0xFF); // Unsigned byte
                    }
                    break;
                case FLOAT16:
                    for (int i = 0; i < cols; i++) {
                        short halfFloat = data.getShort(offset + i * 2);
                        result[i] = halfFloatToFloat(halfFloat);
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported vector type: " + type);
            }
            
            return result;
        }
        
        public byte[] getVectorAsByte(int index) {
            if (index >= rows) {
                throw new IndexOutOfBoundsException("Vector index " + index + " >= " + rows);
            }

            byte[] result = new byte[cols];
            int offset = index * cols * type.getByteSize();
            
            switch (type) {
                case INT8:
                    for (int i = 0; i < cols; i++) {
                        result[i] = data.get(offset + i);
                    }
                    break;
                case FLOAT32:
                    // Convert float to byte for I8 quantization
                    for (int i = 0; i < cols; i++) {
                        float value = data.getFloat(offset + i * Float.BYTES);
                        result[i] = (byte) Math.round(value * 127.0f);
                    }
                    break;
                default:
                    // Convert other types to byte via float
                    float[] floatVector = getVectorAsFloat(index);
                    for (int i = 0; i < cols; i++) {
                        result[i] = (byte) Math.round(floatVector[i] * 127.0f);
                    }
                    break;
            }
            
            return result;
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
        
        logger.info("Loading vector file: {} (type: {})", filePath, type);
        
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // Read header (8 bytes: rows and columns as 32-bit integers)
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();
            
            int totalRows = header.getInt();
            int cols = header.getInt();
            
            logger.info("Dataset dimensions: {} x {} ({})", totalRows, cols, type);
            
            int actualRows = maxRows > 0 ? Math.min(maxRows, totalRows - startRow) : totalRows - startRow;
            if (startRow >= totalRows) {
                throw new IndexOutOfBoundsException("Start row " + startRow + " >= total rows " + totalRows);
            }
            
            // Calculate data size and offset
            long vectorDataSize = (long) totalRows * cols * type.getByteSize();
            long startOffset = 8 + (long) startRow * cols * type.getByteSize();
            long readSize = (long) actualRows * cols * type.getByteSize();
            
            // Read vector data
            channel.position(startOffset);
            ByteBuffer data = ByteBuffer.allocate((int) readSize).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(data);
            data.flip();
            
            logger.info("Loaded {} vectors starting from row {}", actualRows, startRow);
            return new VectorDataset(actualRows, cols, type, data);
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

        public String getPath() { return path; }
        public int getRows() { return rows; }
        public int getCols() { return cols; }
        public VectorType getType() { return type; }
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
}