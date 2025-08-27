# Use official Apache Spark image as base
FROM apache/spark:v3.5.1-scala2.12-java17-python3-r-ubuntu

# Set working directory
WORKDIR /opt/spark/work-dir

# Copy the compiled JAR
COPY build/libs/USearchSpark-1.0-SNAPSHOT.jar /opt/spark/examples/jars/usearch-spark.jar

# Copy dependencies (if any)
COPY build/libs/*.jar /opt/spark/examples/jars/

# Set the user to spark (for security)
USER spark

# Default command
CMD ["spark-submit", "--class", "com.ashvardanian.USearchSpark", "/opt/spark/examples/jars/usearch-spark.jar"]