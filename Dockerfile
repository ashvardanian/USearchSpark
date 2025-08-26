FROM apache/spark:3.5.3-scala2.12-java17-python3-ubuntu

# Switch to root to install additional packages  
USER root

# Java 17 is already installed in the base image  
ENV JAVA_HOME=/opt/java/openjdk

# Create working directory
WORKDIR /app

# Copy the built JAR and dependencies
COPY build/libs/USearchSpark-0.1.0-SNAPSHOT.jar /app/
COPY lib/usearch-2.19.9.jar /app/lib/

# Create output directory and set permissions
RUN mkdir -p /output && chmod -R 777 /output && chmod -R 755 /app

# Run as root to avoid permission issues
USER root

# Default command
CMD ["/opt/spark/bin/spark-submit", "--help"]