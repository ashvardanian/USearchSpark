#!/bin/bash

# Configuration for Spark 4.0
SPARK_VERSION="4.0.0-preview2"
CORES_PER_WORKER=8
MEMORY_PER_WORKER="16g"
TOTAL_CORES=$(nproc)
NUM_WORKERS=$((TOTAL_CORES / CORES_PER_WORKER))

# Install Spark in /opt to avoid polluting the project directory
SPARK_INSTALL_DIR="/opt"
SPARK_HOME="$SPARK_INSTALL_DIR/spark-$SPARK_VERSION-bin-hadoop3"

echo "🚀 Setting up Spark 4.0 Cluster"
echo "System has $TOTAL_CORES cores"
echo "Will create $NUM_WORKERS workers with $CORES_PER_WORKER cores each"
echo "📁 Installing Spark to $SPARK_HOME"

# Download and install Spark 4 if not present
if [ ! -d "$SPARK_HOME" ]; then
    echo "📦 Downloading Apache Spark 4.0..."
    cd /tmp
    wget -q "https://archive.apache.org/dist/spark/spark-4.0.0-preview2/spark-$SPARK_VERSION-bin-hadoop3.tgz"
    if [ $? -eq 0 ]; then
        echo "✅ Download successful, extracting..."
        sudo tar -xzf "spark-$SPARK_VERSION-bin-hadoop3.tgz" -C "$SPARK_INSTALL_DIR"
        rm "spark-$SPARK_VERSION-bin-hadoop3.tgz"
        sudo chown -R $USER:$USER "$SPARK_HOME"
    else
        echo "❌ Download failed, trying alternative URL..."
        wget -q "https://dlcdn.apache.org/spark/spark-4.0.0-preview2/spark-$SPARK_VERSION-bin-hadoop3.tgz"
        sudo tar -xzf "spark-$SPARK_VERSION-bin-hadoop3.tgz" -C "$SPARK_INSTALL_DIR"
        rm "spark-$SPARK_VERSION-bin-hadoop3.tgz"
        sudo chown -R $USER:$USER "$SPARK_HOME"
    fi
    cd - > /dev/null
fi

export SPARK_HOME="$SPARK_HOME"
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

# Configure Spark
echo "⚙️ Configuring Spark cluster..."
cat > $SPARK_HOME/conf/spark-env.sh << EOF
export SPARK_WORKER_CORES=$CORES_PER_WORKER
export SPARK_WORKER_MEMORY=$MEMORY_PER_WORKER
export SPARK_WORKER_INSTANCES=$NUM_WORKERS
export SPARK_MASTER_HOST=localhost
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080
EOF

# Configure Spark defaults
cat > $SPARK_HOME/conf/spark-defaults.conf << EOF
# Spark 4.0 optimizations
spark.serializer                 org.apache.spark.serializer.KryoSerializer
spark.sql.adaptive.enabled       true
spark.sql.adaptive.coalescePartitions.enabled    true
spark.executor.memory            ${MEMORY_PER_WORKER}
spark.executor.cores             ${CORES_PER_WORKER}
spark.executor.instances         ${NUM_WORKERS}

# Network and shuffle optimizations
spark.network.timeout           300s
spark.executor.heartbeatInterval 10s
spark.sql.shuffle.partitions     $((NUM_WORKERS * CORES_PER_WORKER))

# Vector processing optimizations  
spark.sql.execution.arrow.pyspark.enabled       true
spark.sql.execution.arrow.maxRecordsPerBatch    10000
EOF

# Stop any existing Spark services
echo "🛑 Stopping any existing Spark services..."
$SPARK_HOME/sbin/stop-all.sh 2>/dev/null

sleep 2

# Start the cluster
echo "🎯 Starting Spark Master..."
$SPARK_HOME/sbin/start-master.sh

sleep 3

echo "👥 Starting $NUM_WORKERS Spark Workers..."
for i in $(seq 1 $NUM_WORKERS); do
    SPARK_WORKER_WEBUI_PORT=$((8081 + i - 1)) \
    $SPARK_HOME/sbin/start-worker.sh spark://localhost:7077
    echo "  ✅ Started worker $i with $CORES_PER_WORKER cores on port $((8081 + i - 1))"
    sleep 1
done

echo ""
echo "🎉 Spark 4.0 cluster is running!"
echo "📊 Master UI: http://localhost:8080"
echo "🔧 Workers: $NUM_WORKERS x $CORES_PER_WORKER cores = $((NUM_WORKERS * CORES_PER_WORKER)) total cores"
echo ""
echo "🚀 To run your USearchSpark benchmark on this cluster:"
echo "  gradle run --args=\"--master spark://localhost:7077 --mode distributed <dataset-name>\""
echo ""
echo "📈 Single-node baseline (for comparison):"
echo "  gradle run --args=\"--mode local <dataset-name>\""
echo ""
echo "🛑 To stop the cluster:"
echo "  $SPARK_HOME/sbin/stop-all.sh"
echo ""

# Export environment for current session
echo "💡 Run this to set environment variables in your current shell:"
echo "export SPARK_HOME=$SPARK_HOME"
echo "export PATH=\$SPARK_HOME/bin:\$SPARK_HOME/sbin:\$PATH"