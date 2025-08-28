package com.ashvardanian;

/**
 * Time-based progress logger to avoid excessive output during benchmarking
 */
public class ProgressLogger {
    private final String operation;
    private final long totalItems;
    private final long intervalMs;
    private long lastLogTime;
    private final long startTime;

    public ProgressLogger(String operation, long totalItems, long intervalMs) {
        this.operation = operation;
        this.totalItems = totalItems;
        this.intervalMs = intervalMs;
        this.startTime = System.currentTimeMillis();
        this.lastLogTime = startTime;
        
        System.out.println("🔄 " + operation + "...");
    }

    public ProgressLogger(String operation, long totalItems) {
        this(operation, totalItems, 5000); // Default 5 second intervals
    }

    public void update(long currentItem) {
        long now = System.currentTimeMillis();
        
        if (now - lastLogTime >= intervalMs) {
            double progress = (double) currentItem / totalItems * 100.0;
            long elapsed = now - startTime;
            double rate = currentItem / (elapsed / 1000.0);
            
            String rateUnit = operation.toLowerCase().contains("search") ? "QPS" : "IPS";
            System.out.println("   Progress: " + String.format("%.1f%% (%,d/%,d, %,.0f %s)", 
                progress, currentItem, totalItems, rate, rateUnit));
            
            lastLogTime = now;
        }
    }

    public void complete(long actualItems) {
        long elapsed = System.currentTimeMillis() - startTime;
        double rate = actualItems / (elapsed / 1000.0);
        String rateUnit = operation.toLowerCase().contains("search") ? "QPS" : "IPS";
        
        System.out.println("   ✅ " + operation + " completed: " + 
            String.format("%,d items, %,.0f %s", actualItems, rate, rateUnit));
    }
}