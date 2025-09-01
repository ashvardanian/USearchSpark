package com.ashvardanian;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.TieredMergePolicy;

/**
 * A custom merge policy that stops merging once the segment count drops to 2x the thread count. This maintains optimal
 * parallelism for search operations while reducing excessive merge operations.
 */
public class ThreadOptimizedMergePolicy extends FilterMergePolicy {
    private final int targetSegmentCount;
    private final TieredMergePolicy delegate;

    /**
     * Creates a new ThreadOptimizedMergePolicy
     *
     * @param threadCount
     *            The number of threads available for searching/indexing
     */
    public ThreadOptimizedMergePolicy(int threadCount) {
        super(new TieredMergePolicy());
        this.targetSegmentCount = threadCount * 2; // 2x thread count for optimal parallelism
        this.delegate = (TieredMergePolicy)in;

        // Configure delegate for efficient merging when needed
        delegate.setMaxMergedSegmentMB(Long.MAX_VALUE / (1024.0 * 1024.0)); // Maximum allowed value
        delegate.setSegmentsPerTier(3.0); // Aggressive merging to reach target faster
        delegate.setMaxMergeAtOnce(5); // Merge more segments at once
        delegate.setFloorSegmentMB(1024); // 1GB floor to avoid tiny segments
    }

    @Override
    public MergeSpecification findMerges(MergeTrigger trigger, SegmentInfos segmentInfos, MergeContext mergeContext)
            throws IOException {

        int currentSegmentCount = segmentInfos.size();

        // Stop merging if we're at or below our target
        if (currentSegmentCount <= targetSegmentCount) {
            System.out.println(String.format("🛑 Stopping merges: %d segments <= %d target (2x %d threads)",
                    currentSegmentCount, targetSegmentCount, targetSegmentCount / 2));
            return null; // No merges needed
        }

        // Still too many segments - delegate to TieredMergePolicy
        System.out.println(String.format("🔄 Continuing merges: %d segments > %d target", currentSegmentCount,
                targetSegmentCount));
        return delegate.findMerges(trigger, segmentInfos, mergeContext);
    }

    @Override
    public MergeSpecification findForcedMerges(SegmentInfos segmentInfos, int maxSegmentCount,
            Map<SegmentCommitInfo, Boolean> segmentsToMerge, MergeContext mergeContext) throws IOException {
        // Respect forced merges but still apply our minimum threshold
        int effectiveMaxSegments = Math.max(maxSegmentCount, targetSegmentCount);
        return delegate.findForcedMerges(segmentInfos, effectiveMaxSegments, segmentsToMerge, mergeContext);
    }

    @Override
    public MergeSpecification findForcedDeletesMerges(SegmentInfos segmentInfos, MergeContext mergeContext)
            throws IOException {
        // Only merge for deletes if we're above target segment count
        if (segmentInfos.size() <= targetSegmentCount) {
            return null;
        }
        return delegate.findForcedDeletesMerges(segmentInfos, mergeContext);
    }

    /**
     * Get the target segment count
     */
    public int getTargetSegmentCount() {
        return targetSegmentCount;
    }
}
