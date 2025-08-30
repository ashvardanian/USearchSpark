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
    private final KnnVectorsFormat knn;

    public CustomHnswCodec(int m, int beamWidth) {
        super("customHNSW", Codec.getDefault());
        this.m = m;
        this.beamWidth = beamWidth;
        this.knn = createHnswFormat(m, beamWidth);
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return knn;
    }

    private static KnnVectorsFormat createHnswFormat(int m, int beamWidth) {
        String[] candidates = new String[]{
                // Lucene 10.x
                "org.apache.lucene.codecs.lucene1010.Lucene1010HnswVectorsFormat",
                // Lucene 9.x fallbacks
                "org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat",
                "org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsFormat",
                "org.apache.lucene.codecs.hnsw.HnswVectorsFormat"};
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
