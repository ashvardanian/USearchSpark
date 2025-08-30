package com.ashvardanian;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Thread-safe progress logger optimized for billion-scale parallel operations Uses atomic increment with periodic
 * time-based reporting to minimize overhead
 */
public class ProgressLogger {
    private final String operation;
    private final long totalItems;
    private final long intervalMs;
    private final long startTime;
    private final AtomicLong lastLogTime = new AtomicLong(0);
    private final AtomicLong currentProgress = new AtomicLong(0);
    private final int checkInterval;

    public ProgressLogger(String operation, long totalItems, long intervalMs) {
        this.operation = operation;
        this.totalItems = totalItems;
        this.intervalMs = intervalMs;
        this.startTime = System.currentTimeMillis();

        // Dynamic check interval: fewer time checks for larger datasets
        if (totalItems > 100_000_000) {
            this.checkInterval = 50_000; // Check time every 50K items for 100M+ items
        } else if (totalItems > 10_000_000) {
            this.checkInterval = 10_000; // Check time every 10K items for 10M+ items
        } else if (totalItems > 1_000_000) {
            this.checkInterval = 5_000; // Check time every 5K items for 1M+ items
        } else {
            this.checkInterval = 1_000; // Check time every 1K items for smaller datasets
        }

        System.out.println("🔄 " + operation + "...");
    }

    public ProgressLogger(String operation, long totalItems) {
        this(operation, totalItems, 5000); // Default 5 second intervals
    }

    /**
     * Atomic increment - optimized for billion-scale parallel processing Only checks time periodically to minimize
     * System.currentTimeMillis() overhead
     */
    public void increment() {
        long newProgress = currentProgress.incrementAndGet();

        // Only check time every N operations to reduce System.currentTimeMillis()
        // overhead
        if (newProgress % checkInterval == 0) {
            long now = System.currentTimeMillis();
            long lastTime = lastLogTime.get();

            // Atomic time-check update - only one thread logs per interval
            if (now - lastTime >= intervalMs && lastLogTime.compareAndSet(lastTime, now)) {
                double progress = (double) newProgress / totalItems * 100.0;
                long elapsed = now - startTime;
                double rate = elapsed > 0 ? (newProgress * 1000.0) / elapsed : 0;

                String rateUnit = operation.toLowerCase().contains("search") ? "QPS" : "IPS";
                System.out.println("   Progress: " + String.format("%.1f%% (%,d/%,d, %,.0f %s)", progress, newProgress,
                        totalItems, rate, rateUnit));
            }
        }
    }

    public void complete(long actualItems) {
        long elapsed = System.currentTimeMillis() - startTime;
        double rate = elapsed > 0 ? (actualItems * 1000.0) / elapsed : 0;
        String rateUnit = operation.toLowerCase().contains("search") ? "QPS" : "IPS";

        System.out.println("   ✅ " + operation + " completed: "
                + String.format("%,d items, %,.0f %s", actualItems, rate, rateUnit));
    }
}
