package com.ashvardanian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BenchmarkConfig {
    private static final Logger logger = LoggerFactory.getLogger(BenchmarkConfig.class);

    public enum BenchmarkMode {
        LOCAL("local"),
        CLUSTER("cluster");

        private final String name;

        BenchmarkMode(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public static BenchmarkMode fromString(String mode) {
            for (BenchmarkMode m : values()) {
                if (m.name.equalsIgnoreCase(mode)) {
                    return m;
                }
            }
            throw new IllegalArgumentException("Unknown benchmark mode: " + mode);
        }
    }

    public enum Precision {
        F32("f32"),
        F16("f16"),
        BF16("bf16"),
        I8("i8");

        private final String name;

        Precision(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public static Precision fromString(String precision) {
            for (Precision p : values()) {
                if (p.name.equalsIgnoreCase(precision)) {
                    return p;
                }
            }
            throw new IllegalArgumentException("Unknown precision: " + precision);
        }
    }

    private final String datasetName;
    private final String outputPath;
    private final BenchmarkMode mode;
    private final List<Precision> precisions;
    private final Map<String, String> additionalConfig;

    // Default benchmark parameters
    private int numQueries = 10000;
    private int[] kValues = { 1, 10, 100 };
    private boolean includeIndexingTime = true;
    private boolean includeMemoryUsage = true;
    private long maxVectors = -1L; // -1 means use all vectors

    public BenchmarkConfig(String datasetName, String outputPath, BenchmarkMode mode,
            List<Precision> precisions, Map<String, String> additionalConfig) {
        this.datasetName = datasetName;
        this.outputPath = outputPath;
        this.mode = mode;
        this.precisions = Collections.unmodifiableList(new ArrayList<>(precisions));
        this.additionalConfig = Collections.unmodifiableMap(new HashMap<>(additionalConfig));

        // Parse additional configuration
        parseAdditionalConfig();
    }

    private void parseAdditionalConfig() {
        if (additionalConfig.containsKey("numQueries")) {
            this.numQueries = Integer.parseInt(additionalConfig.get("numQueries"));
        }
        if (additionalConfig.containsKey("kValues")) {
            String[] kValuesStr = additionalConfig.get("kValues").split(",");
            this.kValues = Arrays.stream(kValuesStr)
                    .mapToInt(Integer::parseInt)
                    .toArray();
        }
        if (additionalConfig.containsKey("includeIndexingTime")) {
            this.includeIndexingTime = Boolean.parseBoolean(additionalConfig.get("includeIndexingTime"));
        }
        if (additionalConfig.containsKey("includeMemoryUsage")) {
            this.includeMemoryUsage = Boolean.parseBoolean(additionalConfig.get("includeMemoryUsage"));
        }
        if (additionalConfig.containsKey("maxVectors")) {
            this.maxVectors = Long.parseLong(additionalConfig.get("maxVectors"));
        }
    }

    public static BenchmarkConfig parseArgs(String[] args) {
        if (args.length < 1) {
            throw new IllegalArgumentException("Dataset name or config file required");
        }

        String datasetName = args[0];
        BenchmarkMode mode = BenchmarkMode.LOCAL;
        String outputPath = "benchmark_results";
        List<Precision> precisions = Arrays.asList(Precision.F32, Precision.F16, Precision.BF16, Precision.I8);
        Map<String, String> additionalConfig = new HashMap<>();

        // Parse arguments
        for (int i = 1; i < args.length; i++) {
            String arg = args[i];

            if ("--mode".equals(arg) && i + 1 < args.length) {
                mode = BenchmarkMode.fromString(args[++i]);
            } else if ("--output".equals(arg) && i + 1 < args.length) {
                outputPath = args[++i];
            } else if ("--precision".equals(arg) && i + 1 < args.length) {
                String[] precisionStrs = args[++i].split(",");
                precisions = new ArrayList<>();
                for (String p : precisionStrs) {
                    precisions.add(Precision.fromString(p.trim()));
                }
            } else if ("--queries".equals(arg) && i + 1 < args.length) {
                additionalConfig.put("numQueries", args[++i]);
            } else if ("--k-values".equals(arg) && i + 1 < args.length) {
                additionalConfig.put("kValues", args[++i]);
            } else if ("--no-indexing-time".equals(arg)) {
                additionalConfig.put("includeIndexingTime", "false");
            } else if ("--no-memory-usage".equals(arg)) {
                additionalConfig.put("includeMemoryUsage", "false");
            } else if ("--max-vectors".equals(arg) && i + 1 < args.length) {
                additionalConfig.put("maxVectors", args[++i]);
            } else if (arg.startsWith("--")) {
                throw new IllegalArgumentException("Unknown option: " + arg);
            }
        }

        logger.info("Parsed benchmark configuration:");
        logger.info("  Dataset: {}", datasetName);
        logger.info("  Mode: {}", mode);
        logger.info("  Output: {}", outputPath);
        logger.info("  Precisions: {}", precisions);

        return new BenchmarkConfig(datasetName, outputPath, mode, precisions, additionalConfig);
    }

    // Getters
    public String getDatasetName() {
        return datasetName;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public BenchmarkMode getMode() {
        return mode;
    }

    public List<Precision> getPrecisions() {
        return precisions;
    }

    public List<String> getPrecisionNames() {
        return precisions.stream().map(Precision::getName).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }

    public int getNumQueries() {
        return numQueries;
    }

    public int[] getKValues() {
        return kValues;
    }

    public boolean isIncludeIndexingTime() {
        return includeIndexingTime;
    }

    public boolean isIncludeMemoryUsage() {
        return includeMemoryUsage;
    }

    public long getMaxVectors() {
        return maxVectors;
    }

    public String getAdditionalConfig(String key) {
        return additionalConfig.get(key);
    }

    public Map<String, String> getAllAdditionalConfig() {
        return additionalConfig;
    }
}