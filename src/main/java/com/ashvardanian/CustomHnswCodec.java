package com.ashvardanian;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;

/**
 * Custom codec that attempts to configure HNSW parameters (M and beamWidth) if supported by the Lucene version present
 * at runtime. Falls back to the default KNN vectors format if parameterization isn't available.
 */
public class CustomHnswCodec extends FilterCodec {
    private final int m;
    private final int beamWidth;
    private volatile KnnVectorsFormat knn; // Lazy initialization

    // Default constructor required for SPI
    public CustomHnswCodec() {
        this(32, 512); // Use default values
    }

    public CustomHnswCodec(int m, int beamWidth) {
        // Use Lucene101Codec as base (Lucene 10.x) to avoid circular dependency during SPI loading
        super("customHNSW", getLucene101CodecSafely());
        this.m = m;
        this.beamWidth = beamWidth;
        // Don't initialize knn in constructor to avoid circular dependency
    }

    private static org.apache.lucene.codecs.Codec getLucene101CodecSafely() {
        try {
            return new org.apache.lucene.codecs.lucene101.Lucene101Codec();
        } catch (Exception e) {
            // Fallback to SimpleText if Lucene101 not available
            return new org.apache.lucene.codecs.simpletext.SimpleTextCodec();
        }
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        if (knn == null) {
            synchronized (this) {
                if (knn == null) {
                    knn = createHnswFormat(m, beamWidth);
                }
            }
        }
        return knn;
    }

    private static KnnVectorsFormat createHnswFormat(int m, int beamWidth) {
        String[] candidates = new String[]{
                // Lucene 10.x (current) - uses Lucene99 format
                "org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat",
                // Lucene 10.x with scalar quantization
                "org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat",
                // Lucene 10.x with binary quantization
                "org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat"};
        for (String cname : candidates) {
            try {
                Class<?> cls = Class.forName(cname);
                try {
                    var ctor = cls.getConstructor(int.class, int.class);
                    Object inst = ctor.newInstance(m, beamWidth);
                    System.out.println(
                            "🔧 CustomHnswCodec: Using " + cname + " with M=" + m + ", beamWidth=" + beamWidth);
                    return (KnnVectorsFormat) inst;
                } catch (NoSuchMethodException e) {
                    // Fallback: default constructor, parameters not supported
                    Object inst = cls.getDeclaredConstructor().newInstance();
                    System.out.println("⚠️  CustomHnswCodec: Using " + cname
                            + " with DEFAULT parameters (M/beamWidth not supported)");
                    return (KnnVectorsFormat) inst;
                }
            } catch (Throwable ignore) {
                // Try next
            }
        }
        // Final fallback to whatever default codec provides
        System.out.println("❌ CustomHnswCodec: No HNSW implementation found, using default codec's KnnVectorsFormat");
        return Codec.getDefault().knnVectorsFormat();
    }
}
