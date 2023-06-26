# MLOps Pipeline IMDB

[![Continuous Integration](https://github.com/gao-hongnan/mlops-pipeline-imdb/actions/workflows/continuous_integration.yaml/badge.svg)](https://github.com/gao-hongnan/mlops-pipeline-imdb/actions/workflows/continuous_integration.yaml)

This is a toy project to illustrate how to build an end-to-end DataOps and MLOps
pipeline for a movie recommendation system using different modern data and
machine learning technologies.

As a starter, we will use minimal modern tech stacks to build everything from
first principles. Then we will gradually replace the minimal tech stacks with
more sophisticated tech stacks.

For example, we will not use Airflow to orchestrate the pipeline. Instead, we
will use a simple python class named `Orchestrator` to create a Directed Acyclic
Graph (DAG) to orchestrate the pipeline. Then we will replace the `Orchestrator`
with Airflow.

## Problem Statement

To initiate the project, we will start with a simple binary classifier as the
first version, instead of a fully-fledged recommendation system. The function of
this system will be straightforward. Given a movie title and its genre as
inputs, the system will generate an output indicating if the user is likely to
enjoy the movie or not.

In this context, the prediction will be either $0$ or $1$, symbolizing whether
the user will appreciate the movie. The user ratings in the training data vary
from $1$ to $10$. However, for simplification, these ratings are rounded to $0$
or $1$ by establishing a decision threshold at $5$. Therefore, the final product
will, based on a provided movie title and genre, yield a binary output.

## Workflow

...

## Kubernetes

### Setting up

We follow
[the guide](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/) here
to set up `kubectl` on Mac.

```bash
brew install kubectl
```

### Build Docker Image Locally

You need to build using `linux/amd64` platform or else GKE will encounter an
[**error**](https://stackoverflow.com/questions/42494853/standard-init-linux-go178-exec-user-process-caused-exec-format-error).

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
--platform=linux/amd64 \
-f pipeline-training/Dockerfile \
-t pipeline-training:$GIT_COMMIT_HASH \
.
```

### Run Docker Image Locally

```bash
docker run -it \
    --env PROJECT_ID="${PROJECT_ID}" \
    --env GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS}" \
    --env GCS_BUCKET_NAME="${GCS_BUCKET_NAME}" \
    --env GCS_BUCKET_PROJECT_NAME="${GCS_BUCKET_PROJECT_NAME}" \
    --env BIGQUERY_RAW_DATASET="${BIGQUERY_RAW_DATASET}" \
    --env BIGQUERY_RAW_TABLE_NAME="${BIGQUERY_RAW_TABLE_NAME}" \
    pipeline-training:${GIT_COMMIT_HASH}

docker run -it \
    --env PROJECT_ID="gao-hongnan" \
    --env GOOGLE_APPLICATION_CREDENTIALS="/pipeline-training/gcp-storage-service-account.json" \
    --env GCS_BUCKET_NAME="gaohn" \
    --env GCS_BUCKET_PROJECT_NAME="imdb" \
    --env BIGQUERY_RAW_DATASET=imdb_dbt_filtered_movies_incremental \
    --env BIGQUERY_RAW_TABLE_NAME=filtered_movies_incremental \
    pipeline-training:${GIT_COMMIT_HASH}

docker exec -it <CONTAINER> /bin/bash
```

### Build Docker Image to Push to Container Registry

Check `gar_docker_setup` in my `common-utils` on how to set up a container
registry in GCP.

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export PROJECT_ID="gao-hongnan" && \
export REPO_NAME="imdb-repo" && \
export APP_NAME="mlops-pipeline-imdb" && \
export REGION="us-west2" && \
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
-f pipeline-training/Dockerfile \
--platform=linux/amd64 \
-t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}:${GIT_COMMIT_HASH}" \
.
```

### Push Docker Image to Artifacts Registry

```bash
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}:${GIT_COMMIT_HASH}" && \
echo "Successfully pushed ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}:${GIT_COMMIT_HASH}"
```

### Deploy Docker Image from Artifacts Registry to Google Kubernetes Engine

Note you do not need to provide the `GIT_COMMIT_HASH` as the image is already
tagged with the `GIT_COMMIT_HASH`.

- Set up `Expose` to false since this is not a web app.

## Pipeline Training

```bash
export PYTHONPATH="${PYTHONPATH}:/Users/gaohn/gaohn/mlops-pipeline-imdb/pipeline-training"
```

```bash
python pipeline-training/pipeline_training/data_extraction/extract.py
python pipeline-training/pipeline_training/data_validation/validate.py
rm -rf /Users/gaohn/gaohn/end2end-movie-recommender-system/pipeline-training/data
python pipeline-training/pipeline_training/data_loading/load.py
# test production dvc pull
rm /Users/gaohn/gaohn/end2end-movie-recommender-system/pipeline-training/data/raw/filtered_movies_incremental.csv
python pipeline-training/pipeline_training/pipeline_production.py
```

```bash
mkdir -p pipeline-training && \
mkdir -p conf && \
touch conf/__init__.py && \
mkdir -p tests && \
touch tests/__init__.py && \
touch requirements.txt requirements_dev.txt && \
touch README.md && \
touch pyproject.toml && \
```

```bash
cd pipeline-training && \
mkdir -p pipeline_training && \
cd pipeline_training && \
touch __init__.py && \
mkdir -p data_extraction data_validation data_preparation && \
touch data_extraction/__init__.py data_validation/__init__.py data_preparation/__init__.py && \
mkdir -p model_training model_evaluation model_validation && \
touch model_training/__init__.py model_evaluation/__init__.py model_validation/__init__.py && \
mkdir -p utils && \
touch utils/__init__.py && \
```

Create data and store dir.

### Setting up Virtual Environment

We will use the training pipeline as an example to illustrate how to set up the
virtual environment.

```bash
cd pipeline-training && \
curl -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh
```

```bash
bash make_venv.sh --pyproject --dev
```

## Pipeline App

```bash
cd pipeline-app && \
mkdir -p api frontend monitoring && \
```

### Setting up Folder Structure

```bash
mkdir -p pipeline-app && \
...
```

To fill in

### Setting up Virtual Environment

...

## Flow

1. Extract from Data Warehouse where the table from the warehouse is considered
   `raw`, `staging` and `immutable`. This table is most likely a result of a
   typical `ETL` or `ELT` process where the data is extracted from a source
   system such as an API, then transformed and loaded into a data warehouse as
   well as a data lake. This data is for downstream consumption and is
   considered immutable.
2. Validate the data. This is to ensure that the data has expected values and is
   in the expected format. See GCP's CICD blog.
3. Load the data to local and Data Lake. At this stage, if you do `tree data`
   you should see the following:

    ```tree
    .
    ├── raw/
    │   └── <table_name>.csv
    ```

    and of similar structure for the data lake.

4. **Data Versioning Alert**: One super important step is to hash the data.
   Let's see the example.

    - I loaded the data `data.csv` from the data warehouse to the local folder
      `data/raw`.
    - I then hashed the data using `dvc add data/raw/data.csv`.
    - The output is

        ```json
        {
         "filename": "filtered_movies_incremental.csv",
         "filepath": ".cache/483b93622aea897850c63597e852bb25",
         "hash": "483b93622aea897850c63597e852bb25"
        }
        ```

    - I then added the file to the git repo using
      `git add data/raw/data.csv.dvc`.
    - Now this particular file is tracked by git and dvc.
    - Say the `git` commit hash is `1234567` and the `dvc` commit hash is
      `483b93622aea897850c63597e852bb25`.
    - When i checkout the `git` commit hash `1234567`, the `dvc` commit hash is
      still `483b93622aea897850c63597e852bb25`, allowing me to checkout the data
      at that particular commit hash.

## TODOs

1. Remember to compose all the configs like `Metadata` and `Directories` into
   `Config` for final refactoring.
2. Spend time describing my `Logger` class!
3. Spend time talking about my `DVC` implementation.


