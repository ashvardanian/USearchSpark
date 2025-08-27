package com.ashvardanian;

import java.util.List;
import java.util.Map;

public class BenchmarkResultsTable {
    
    public static void printComparisonTable(Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults,
                                          LuceneBenchmark.BenchmarkResult luceneResult) {
        System.out.println("\n" + "=".repeat(100));
        System.out.println("🏆 VECTOR SEARCH BENCHMARK RESULTS (vs Lucene F32 Baseline)");
        System.out.println("=".repeat(100));
        
        // Table header
        System.out.printf("%-12s | %-9s | %-10s | %-10s | %-8s | %-10s | %-10s | %-12s%n",
            "Implementation", "Precision", "Index (ms)", "Search (ms)", "QPS", "Recall@1", "Recall@10", "vs Baseline");
        System.out.println("-".repeat(100));
        
        // Lucene baseline (reference point)
        System.out.printf("%-12s | %-9s | %,10d | %,10d | %8.0f | %10.4f | %10.4f | %12s%n",
            "Lucene", "F32", 
            luceneResult.getIndexingTimeMs(),
            luceneResult.getSearchTimeMs(),
            luceneResult.getThroughputQPS(),
            luceneResult.getRecallAtK().get(1),
            luceneResult.getRecallAtK().get(10),
            "BASELINE");
        
        System.out.println("-".repeat(100));
        
        // USearch results - all compared to Lucene F32
        List<BenchmarkConfig.Precision> precisions = List.of(
            BenchmarkConfig.Precision.F32,
            BenchmarkConfig.Precision.F16, 
            BenchmarkConfig.Precision.BF16,
            BenchmarkConfig.Precision.I8
        );
        
        for (BenchmarkConfig.Precision precision : precisions) {
            USearchBenchmark.BenchmarkResult result = usearchResults.get(precision);
            if (result != null) {
                // Calculate speedup vs Lucene F32 baseline
                double qpsSpeedup = result.getThroughputQPS() / luceneResult.getThroughputQPS();
                String speedupStr = String.format("%.1fx faster", qpsSpeedup);
                if (qpsSpeedup < 1.0) {
                    speedupStr = String.format("%.1fx slower", 1.0 / qpsSpeedup);
                }
                
                System.out.printf("%-12s | %-9s | %,10d | %,10d | %8.0f | %10.4f | %10.4f | %12s%n",
                    "USearch", precision.getName().toUpperCase(),
                    result.getIndexingTimeMs(),
                    result.getSearchTimeMs(),
                    result.getThroughputQPS(),
                    result.getRecallAtK().get(1),
                    result.getRecallAtK().get(10),
                    speedupStr);
            }
        }
        
        System.out.println("=".repeat(100));
        
        // Performance summary - all vs Lucene F32
        System.out.println("🚀 PERFORMANCE SUMMARY (vs Lucene F32 Baseline):");
        
        for (BenchmarkConfig.Precision precision : precisions) {
            USearchBenchmark.BenchmarkResult result = usearchResults.get(precision);
            if (result != null) {
                double indexSpeedup = (double) luceneResult.getIndexingTimeMs() / result.getIndexingTimeMs();
                double searchSpeedup = (double) luceneResult.getSearchTimeMs() / result.getSearchTimeMs();
                double qpsImprovement = result.getThroughputQPS() / luceneResult.getThroughputQPS();
                
                System.out.printf("   • USearch %s: %.1fx indexing, %.1fx search, %.1fx throughput%n", 
                    precision.getName().toUpperCase(), indexSpeedup, searchSpeedup, qpsImprovement);
            }
        }
        
        // Highlight the best performer
        USearchBenchmark.BenchmarkResult bestResult = null;
        BenchmarkConfig.Precision bestPrecision = null;
        double bestQps = 0;
        
        for (Map.Entry<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> entry : usearchResults.entrySet()) {
            if (entry.getValue().getThroughputQPS() > bestQps) {
                bestQps = entry.getValue().getThroughputQPS();
                bestResult = entry.getValue();
                bestPrecision = entry.getKey();
            }
        }
        
        if (bestResult != null) {
            double bestImprovement = bestResult.getThroughputQPS() / luceneResult.getThroughputQPS();
            System.out.printf("%n🏅 WINNER: USearch %s with %.1fx better throughput than Lucene F32!%n", 
                bestPrecision.getName().toUpperCase(), bestImprovement);
        }
        
        System.out.println();
    }
}