# Use your prebuilt base image
FROM eda-fastapi:latest

COPY hadoop /opt/hadoop

# Set Spark environment
ENV HADOOP_HOME=/opt/hadoop
ENV PATH="$HADOOP_HOME/bin:$PATH"