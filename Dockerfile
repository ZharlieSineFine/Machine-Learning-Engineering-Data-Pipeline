# Apache Airflow 2.6.1 on Python 3.10.
# NOTE: the default `apache/airflow:2.6.1` tag is a Python 3.7 image, too old for
# pyspark 3.5.5 / modern pandas. The -python3.10 variant fixes that.
FROM apache/airflow:2.6.1-python3.10

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java 17 for PySpark; procps for `ps` (Spark needs it); bash for Spark scripts.
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Pipeline Python deps (Airflow comes from the base image).
COPY requirements.txt /requirements.txt
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt
