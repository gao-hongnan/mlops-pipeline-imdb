ARG PYTHON_VERSION=3.9

# Base image
FROM python:${PYTHON_VERSION}-slim-buster as builder

# Set work directory to /pipeline-training
WORKDIR /pipeline-training

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements, to cache them in docker layer
ARG REQUIREMENTS=requirements.txt
ARG REQUIREMENTS_DEV=requirements_dev.txt
COPY ./${REQUIREMENTS} .
COPY ./${REQUIREMENTS_DEV} .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS} && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS_DEV} && \
    pip install -U gaohn-common-utils


# This is the real runner for my app
FROM python:${PYTHON_VERSION}-slim-buster as runner

# Install cron
RUN apt-get update && apt-get install -y cron

# Copy from builder image
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /pipeline-training /pipeline-training

# Set work dir again to the pipeline_training subdirectory
# Set the working directory inside the Docker container
WORKDIR /pipeline-training


ENV PATH="/opt/venv/bin:$PATH"

# Copy the rest of the application
# Copy the rest of the application
COPY .env .env
COPY ./conf /pipeline-training/conf
COPY ./pipeline_training /pipeline-training/pipeline_training
COPY ./gcp-storage-service-account.json /pipeline-training/gcp-storage-service-account.json

CMD ["python", "-m", "pipeline_feature/pipeline.py"]