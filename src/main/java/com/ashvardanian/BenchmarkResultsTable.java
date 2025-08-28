package com.ashvardanian;

import java.util.List;
import java.util.Map;

import de.vandermeer.asciitable.AsciiTable;
import de.vandermeer.skb.interfaces.transformers.textformat.TextAlignment;

public class BenchmarkResultsTable {
    
    private static String formatMemory(long bytes) {
        double mb = bytes / (1024.0 * 1024.0);
        if (mb >= 1024) {
            return String.format("%.1f GB", mb / 1024.0);
        }
        return String.format("%.0f MB", mb);
    }
    
    public static void printComparisonTable(Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults,
                                          LuceneBenchmark.BenchmarkResult luceneResult) {
        System.out.println("\n📊 VECTOR SEARCH BENCHMARK RESULTS 📊");
        System.out.println("═".repeat(85));
        System.out.println();
        
        // Performance metrics table
        System.out.println("🚀 PERFORMANCE METRICS");
        AsciiTable perfTable = new AsciiTable();
        perfTable.addRule();
        perfTable.addRow("Engine", "Precision", "IPS", "QPS", "Memory");
        perfTable.addRule();
        
        // Lucene first as requested
        double luceneIps = luceneResult.getIndexingTimeMs() > 0 ? (1000000.0 * 1000.0) / luceneResult.getIndexingTimeMs() : 0;
        perfTable.addRow("Apache", "F32", 
            String.format("%,.0f", luceneIps),
            String.format("%,.0f", luceneResult.getThroughputQPS()),
            formatMemory(luceneResult.getMemoryUsageBytes()));
        
        // USearch results
        List<BenchmarkConfig.Precision> precisions = List.of(
            BenchmarkConfig.Precision.F32,
            BenchmarkConfig.Precision.F16, 
            BenchmarkConfig.Precision.BF16,
            BenchmarkConfig.Precision.I8
        );
        
        for (BenchmarkConfig.Precision precision : precisions) {
            USearchBenchmark.BenchmarkResult result = usearchResults.get(precision);
            if (result != null) {
                double ips = result.getIndexingTimeMs() > 0 ? (1000000.0 * 1000.0) / result.getIndexingTimeMs() : 0;
                perfTable.addRow("USearch", precision.getName().toUpperCase(),
                    String.format("%,.0f", ips),
                    String.format("%,.0f", result.getThroughputQPS()),
                    formatMemory(result.getMemoryUsageBytes()));
            }
        }
        
        perfTable.addRule();
        perfTable.getContext().setWidth(75);
        perfTable.setPaddingLeftRight(1);
        System.out.println(perfTable.render());
        
        // Recall metrics table
        System.out.println("🎯 RECALL METRICS");
        AsciiTable recallTable = new AsciiTable();
        recallTable.addRule();
        recallTable.addRow("Engine", "Precision", "Recall@1", "Recall@10", "Recall@100");
        recallTable.addRule();
        
        // Lucene first
        recallTable.addRow("Apache", "F32",
            String.format("%.4f", luceneResult.getRecallAtK().get(1)),
            String.format("%.4f", luceneResult.getRecallAtK().get(10)),
            String.format("%.4f", luceneResult.getRecallAtK().getOrDefault(100, 0.0)));
        
        // USearch results
        for (BenchmarkConfig.Precision precision : precisions) {
            USearchBenchmark.BenchmarkResult result = usearchResults.get(precision);
            if (result != null) {
                recallTable.addRow("USearch", precision.getName().toUpperCase(),
                    String.format("%.4f", result.getRecallAtK().get(1)),
                    String.format("%.4f", result.getRecallAtK().get(10)),
                    String.format("%.4f", result.getRecallAtK().getOrDefault(100, 0.0)));
            }
        }
        
        recallTable.addRule();
        recallTable.getContext().setWidth(75);
        recallTable.setPaddingLeftRight(1);
        System.out.println(recallTable.render());
        
        // Find best performer
        USearchBenchmark.BenchmarkResult bestResult = null;
        BenchmarkConfig.Precision bestPrecision = null;
        double bestQps = luceneResult.getThroughputQPS();
        String bestEngine = "Apache";
        
        for (Map.Entry<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> entry : usearchResults.entrySet()) {
            if (entry.getValue().getThroughputQPS() > bestQps) {
                bestQps = entry.getValue().getThroughputQPS();
                bestResult = entry.getValue();
                bestPrecision = entry.getKey();
                bestEngine = "USearch";
            }
        }
        
        if (bestPrecision != null && bestResult != null) {
            double bestRecall = bestResult.getRecallAtK().get(10);
            System.out.printf("🏆 WINNER: %s %s - Best QPS (%,.0f) with %.2f%% recall@10%n", 
                bestEngine, bestPrecision.getName().toUpperCase(), bestQps, bestRecall * 100);
        } else {
            System.out.printf("🏆 WINNER: Apache F32 - Best QPS (%,.0f) with %.2f%% recall@10%n", 
                bestQps, luceneResult.getRecallAtK().get(10) * 100);
        }
        
        System.out.println("💡 IPS = Insertions Per Second (indexing), QPS = Queries Per Second (search)");
        System.out.println();
    }
}