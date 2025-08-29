package com.ashvardanian;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.Lock;
import org.apache.lucene.store.LockFactory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.store.FilterIndexOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A Directory implementation that dynamically grows by adding more ByteBuffersDirectory instances
 * as needed to work around the 4GB limitation of a single ByteBuffersDirectory.
 * 
 * This implementation monitors file sizes and automatically creates new shards when approaching limits.
 */
public class DynamicByteBuffersDirectory extends Directory {
    
    private static final long MAX_SHARD_SIZE = 3L * 1024 * 1024 * 1024; // 3GB per shard (leaving buffer)
    private static final int INITIAL_SHARDS = 4;
    
    private final List<ByteBuffersDirectory> shards;
    private final Map<String, Integer> fileToShard;
    private final Map<Integer, AtomicLong> shardSizes;
    private final AtomicInteger nextShardIndex;
    private final LockFactory lockFactory;
    private final Object growthLock = new Object();
    
    public DynamicByteBuffersDirectory() {
        this.shards = new ArrayList<>();
        this.fileToShard = new ConcurrentHashMap<>();
        this.shardSizes = new ConcurrentHashMap<>();
        this.nextShardIndex = new AtomicInteger(0);
        this.lockFactory = NoLockFactory.INSTANCE;
        
        // Initialize with a few shards
        for (int i = 0; i < INITIAL_SHARDS; i++) {
            addNewShard();
        }
    }
    
    private int addNewShard() {
        synchronized (growthLock) {
            int shardIndex = shards.size();
            shards.add(new ByteBuffersDirectory());
            shardSizes.put(shardIndex, new AtomicLong(0));
            System.out.println("📦 Added new shard #" + shardIndex + " (total: " + (shardIndex + 1) + ")");
            return shardIndex;
        }
    }
    
    private int findOrCreateShardForFile(String name, long estimatedSize) {
        // Check if file already exists
        Integer existingShard = fileToShard.get(name);
        if (existingShard != null) {
            return existingShard;
        }
        
        // Find a shard with enough space
        synchronized (growthLock) {
            for (int i = 0; i < shards.size(); i++) {
                AtomicLong shardSize = shardSizes.get(i);
                if (shardSize.get() + estimatedSize < MAX_SHARD_SIZE) {
                    fileToShard.put(name, i);
                    return i;
                }
            }
            
            // No shard has enough space, create a new one
            int newShardIndex = addNewShard();
            fileToShard.put(name, newShardIndex);
            return newShardIndex;
        }
    }
    
    private ByteBuffersDirectory getShardForRead(String name) throws IOException {
        Integer shardIdx = fileToShard.get(name);
        if (shardIdx == null) {
            // Try to find the file in any shard
            for (int i = 0; i < shards.size(); i++) {
                ByteBuffersDirectory shard = shards.get(i);
                if (shard.fileExists(name)) {
                    fileToShard.put(name, i);
                    return shard;
                }
            }
            throw new IOException("File not found: " + name);
        }
        return shards.get(shardIdx);
    }
    
    @Override
    public String[] listAll() throws IOException {
        Set<String> allFiles = ConcurrentHashMap.newKeySet();
        synchronized (growthLock) {
            for (ByteBuffersDirectory shard : shards) {
                for (String file : shard.listAll()) {
                    allFiles.add(file);
                }
            }
        }
        return allFiles.toArray(new String[0]);
    }
    
    @Override
    public void deleteFile(String name) throws IOException {
        Integer shardIdx = fileToShard.get(name);
        if (shardIdx != null) {
            long fileSize = shards.get(shardIdx).fileLength(name);
            shards.get(shardIdx).deleteFile(name);
            shardSizes.get(shardIdx).addAndGet(-fileSize);
            fileToShard.remove(name);
        }
    }
    
    @Override
    public long fileLength(String name) throws IOException {
        return getShardForRead(name).fileLength(name);
    }
    
    @Override
    public IndexOutput createOutput(String name, IOContext context) throws IOException {
        // Estimate size - for HNSW vectors, this could be large
        long estimatedSize = 100 * 1024 * 1024; // Start with 100MB estimate
        if (name.contains("vec")) {
            estimatedSize = 1024 * 1024 * 1024; // 1GB for vector files
        }
        
        int shardIdx = findOrCreateShardForFile(name, estimatedSize);
        ByteBuffersDirectory shard = shards.get(shardIdx);
        AtomicLong shardSize = shardSizes.get(shardIdx);
        
        // Wrap the output to track size
        IndexOutput delegate = shard.createOutput(name, context);
        return new FilterIndexOutput("SizeTrackingOutput(" + name + ")", delegate) {
            private long bytesWritten = 0;
            
            @Override
            public void writeByte(byte b) throws IOException {
                super.writeByte(b);
                bytesWritten++;
                if (bytesWritten % (100 * 1024 * 1024) == 0) { // Update every 100MB
                    shardSize.addAndGet(100 * 1024 * 1024);
                }
            }
            
            @Override
            public void writeBytes(byte[] b, int offset, int length) throws IOException {
                super.writeBytes(b, offset, length);
                bytesWritten += length;
                if (bytesWritten / (100 * 1024 * 1024) > (bytesWritten - length) / (100 * 1024 * 1024)) {
                    shardSize.addAndGet(100 * 1024 * 1024);
                }
            }
            
            @Override
            public void close() throws IOException {
                super.close();
                // Final size update
                shardSize.set(calculateActualShardSize(shardIdx));
            }
        };
    }
    
    private long calculateActualShardSize(int shardIdx) {
        long total = 0;
        try {
            ByteBuffersDirectory shard = shards.get(shardIdx);
            for (String file : shard.listAll()) {
                total += shard.fileLength(file);
            }
        } catch (IOException e) {
            // Best effort
        }
        return total;
    }
    
    @Override
    public IndexOutput createTempOutput(String prefix, String suffix, IOContext context) throws IOException {
        // Use round-robin for temp files across existing shards
        int shardIdx = nextShardIndex.getAndIncrement() % shards.size();
        return shards.get(shardIdx).createTempOutput(prefix, suffix, context);
    }
    
    @Override
    public void sync(Collection<String> names) throws IOException {
        Map<Integer, Set<String>> shardFiles = new HashMap<>();
        for (String name : names) {
            Integer shardIdx = fileToShard.get(name);
            if (shardIdx != null) {
                shardFiles.computeIfAbsent(shardIdx, k -> ConcurrentHashMap.newKeySet()).add(name);
            }
        }
        
        for (Map.Entry<Integer, Set<String>> entry : shardFiles.entrySet()) {
            shards.get(entry.getKey()).sync(entry.getValue());
        }
    }
    
    @Override
    public void rename(String source, String dest) throws IOException {
        ByteBuffersDirectory shard = getShardForRead(source);
        shard.rename(source, dest);
        Integer shardIdx = fileToShard.remove(source);
        if (shardIdx != null) {
            fileToShard.put(dest, shardIdx);
        }
    }
    
    @Override
    public void syncMetaData() throws IOException {
        synchronized (growthLock) {
            for (ByteBuffersDirectory shard : shards) {
                shard.syncMetaData();
            }
        }
    }
    
    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
        return getShardForRead(name).openInput(name, context);
    }
    
    @Override
    public Lock obtainLock(String name) throws IOException {
        return lockFactory.obtainLock(this, name);
    }
    
    @Override
    public void close() throws IOException {
        synchronized (growthLock) {
            for (ByteBuffersDirectory shard : shards) {
                shard.close();
            }
        }
    }
    
    @Override
    public Set<String> getPendingDeletions() throws IOException {
        Set<String> allPending = ConcurrentHashMap.newKeySet();
        synchronized (growthLock) {
            for (ByteBuffersDirectory shard : shards) {
                allPending.addAll(shard.getPendingDeletions());
            }
        }
        return allPending;
    }
    
    public long estimatedSizeInBytes() {
        long total = 0;
        for (AtomicLong shardSize : shardSizes.values()) {
            total += shardSize.get();
        }
        return total;
    }
    
    @Override
    public String toString() {
        long totalSize = estimatedSizeInBytes();
        return String.format("DynamicByteBuffersDirectory[shards=%d, totalSize=%,d bytes (%.2f GB)]", 
                            shards.size(), totalSize, totalSize / (1024.0 * 1024.0 * 1024.0));
    }
}