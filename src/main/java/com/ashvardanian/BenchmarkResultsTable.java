package com.ashvardanian;

import de.vandermeer.asciitable.AsciiTable;
import java.util.List;
import java.util.Map;

public class BenchmarkResultsTable {

    private static String formatMemory(long bytes) {
        double mb = bytes / (1024.0 * 1024.0);
        if (mb >= 1024) {
            return String.format("%.1f GB", mb / 1024.0);
        }
        return String.format("%.0f MB", mb);
    }

    public static void printComparisonTable(
            Map<BenchmarkConfig.Precision, USearchBenchmark.BenchmarkResult> usearchResults,
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
        double luceneIps = luceneResult.getIndexingTimeMs() > 0
                ? (luceneResult.getNumVectors() * 1000.0) / luceneResult.getIndexingTimeMs()
                : 0;
        perfTable.addRow("Lucene", "F32", String.format("%,.0f", luceneIps),
                String.format("%,.0f", luceneResult.getThroughputQPS()),
                formatMemory(luceneResult.getMemoryUsageBytes()));

        // USearch results
        List<BenchmarkConfig.Precision> precisions = List.of(BenchmarkConfig.Precision.F32,
                BenchmarkConfig.Precision.F16, BenchmarkConfig.Precision.BF16, BenchmarkConfig.Precision.I8);

        for (BenchmarkConfig.Precision precision : precisions) {
            USearchBenchmark.BenchmarkResult result = usearchResults.get(precision);
            if (result != null) {
                double ips = result.getIndexingTimeMs() > 0
                        ? (result.getNumVectors() * 1000.0) / result.getIndexingTimeMs()
                        : 0;
                perfTable.addRow("USearch", precision.getName().toUpperCase(), String.format("%,.0f", ips),
                        String.format("%,.0f", result.getThroughputQPS()), formatMemory(result.getMemoryUsageBytes()));
            }
        }

        perfTable.addRule();
        perfTable.getContext().setWidth(75);
        perfTable.setPaddingLeftRight(1);
        System.out.println(perfTable.render());

        // Recall & NDCG metrics table
        System.out.println("🎯 RECALL & NDCG METRICS");
        AsciiTable metricsTable = new AsciiTable();
        metricsTable.addRule();
        metricsTable.addRow("Engine", "Precision", "Recall@10", "NDCG@10", "Recall@100", "NDCG@100");
        metricsTable.addRule();

        // Lucene first
        metricsTable.addRow("Lucene", "F32", String.format("%.2f%%", luceneResult.getRecallAtK().get(10) * 100.0),
                String.format("%.2f%%", luceneResult.getNDCGAtK().getOrDefault(10, 0.0) * 100.0),
                String.format("%.2f%%", luceneResult.getRecallAtK().getOrDefault(100, 0.0) * 100.0),
                String.format("%.2f%%", luceneResult.getNDCGAtK().getOrDefault(100, 0.0) * 100.0));

        // USearch results
        for (BenchmarkConfig.Precision precision : precisions) {
            USearchBenchmark.BenchmarkResult result = usearchResults.get(precision);
            if (result != null) {
                metricsTable.addRow("USearch", precision.getName().toUpperCase(),
                        String.format("%.2f%%", result.getRecallAtK().get(10) * 100.0),
                        String.format("%.2f%%", result.getNDCGAtK().getOrDefault(10, 0.0) * 100.0),
                        String.format("%.2f%%", result.getRecallAtK().getOrDefault(100, 0.0) * 100.0),
                        String.format("%.2f%%", result.getNDCGAtK().getOrDefault(100, 0.0) * 100.0));
            }
        }

        metricsTable.addRule();
        metricsTable.getContext().setWidth(85);
        metricsTable.setPaddingLeftRight(1);
        System.out.println(metricsTable.render());
        System.out.println();
    }
}
