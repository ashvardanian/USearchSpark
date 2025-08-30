package com.ashvardanian;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95Codec;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsFormat;

/**
 * Custom Lucene codec to configure HNSW parameters (M and beamWidth) per index. Increasing these generally improves
 * recall at the cost of memory and indexing time.
 */
public class CustomHnswCodec extends Lucene95Codec {
    private final int maxConn;
    private final int beamWidth;

    public CustomHnswCodec(int maxConn, int beamWidth) {
        super();
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
    }

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return new Lucene95HnswVectorsFormat(maxConn, beamWidth);
    }
}
