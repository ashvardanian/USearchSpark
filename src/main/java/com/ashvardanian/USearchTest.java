package com.ashvardanian;

import cloud.unum.usearch.Index;

public class USearchTest {
    public static void main(String[] args) {
        try {
            System.out.println("Testing USearch native library loading...");

            // Try to create a simple index to test native loading
            Index index = new Index.Config().metric(Index.Metric.COSINE).quantization(Index.Quantization.FLOAT32)
                    .dimensions(128).capacity(1000).connectivity(16).expansion_add(128).expansion_search(64).build();

            System.out.println("✅ USearch native library loaded successfully!");

            // Test basic operations
            float[] testVector = new float[128];
            for (int i = 0; i < 128; i++) {
                testVector[i] = (float) Math.random();
            }

            index.add(1, testVector);
            long[] results = index.search(testVector, 1);

            System.out.println("✅ Basic USearch operations work!");
            System.out.println("Search result: " + java.util.Arrays.toString(results));

            index.close();

        } catch (UnsatisfiedLinkError e) {
            System.err.println("❌ Native library loading failed: " + e.getMessage());
            System.err.println(
                    "This likely means the USearch JAR doesn't have compatible native libraries for your platform.");
            System.err.println("Platform info:");
            System.err.println("  OS: " + System.getProperty("os.name"));
            System.err.println("  Arch: " + System.getProperty("os.arch"));
            System.err.println("  Java version: " + System.getProperty("java.version"));
        } catch (Exception e) {
            System.err.println("❌ Other error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
