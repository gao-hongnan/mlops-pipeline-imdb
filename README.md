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

## Reproducibility

To ensure that your machine learning experiments are reproducible, you should
keep track of the following components:

1. **Code**
2. **Data**
3. **Model config, artifacts and metadata**

### 1. Code versioning

Use a version control system like **Git** to keep track of your codebase. Git
allows you to track changes in your code over time and manage different
versions. To log the exact commit hash of your codebase when logging your MLflow
run, you can use the following code snippet:

```python
import subprocess

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
)
mlflow.log_param("commit_hash", commit_hash)
```

By logging the commit hash, you can always refer back to the exact version of
the code used for a specific run, ensuring reproducibility.

### 2. Data versioning

For data versioning, you can use a tool like **DVC (Data Version Control)**. DVC
is designed to handle large data files, models, and metrics, and it integrates
well with Git. DVC helps you track changes in your data files and manage
different versions.

When you start a new MLflow run, log the DVC version or metadata of the input
data used in the experiment. This way, you can always retrieve the exact version
of the data used for a specific run, ensuring reproducibility.

See:

- [Data Management Tutorial](https://dvc.org/doc/start/data-management)

Important points:

- gitignore will be created automatically in data folder once you dvc add.
- After successfully pushing the data to remote, how do you "retrieve them"?
- If you are in the same repository, you can just pull the data from remote.

Yes, the idea is to use dvc checkout to switch between different versions of
your data files, as tracked by DVC. When you use dvc checkout, you provide a Git
commit hash or tag. DVC will then update your working directory with the data
files that were tracked at that specific Git commit.

Here are the steps to use dvc checkout with a Git commit hash:

Make sure you have the latest version of your repository and DVC remote by
running git pull and dvc pull.

Switch to the desired Git commit by running git checkout `<commit-hash>`.

Run dvc checkout to update your data files to the version tracked at the
specified commit.

Remember that dvc checkout only updates the data files tracked by DVC. To switch
between code versions, you'll still need to use git checkout.

```bash
git checkout <commit_hash>
dvc checkout # in this commit hash
dvc pull
```

### 3. Model artifacts and metadata

You have already logged the artifacts (model, vectorizer, config, log files)
using `mlflow.log_artifact()`. You can also log additional metadata related to
the artifacts as you have done with additional_metadata. This should be
sufficient for keeping track of the artifacts associated with each run.

By combining code versioning with Git, data versioning with DVC, and logging
artifacts and metadata with MLflow, you can ensure that your machine learning
experiments are reproducible. This makes it easier to share your work,
collaborate with others, and build upon your experiments over time.

### Recovering a run

1. Check the commit hashes for the code and data used in the run.
2. Checkout the code and data versions using the commit hashes.

```bash
git checkout <commit_hash>
pip install -r requirements.txt
python main.py train
# once done
git checkout main
```

## Experiment Tracking

### MLFlow Remote Tracking Server

Mirroring
[my MLFlow example's README](https://github.com/gao-hongnan/common-utils/tree/main/examples/containerization/docker/mlflow)
as well as my MLOps
[**documentation**](https://gao-hongnan.github.io/gaohn-mlops-docs/).

### Method 1. GCP VM

```bash
gcloud compute ssh --zone "asia-southeast1-a" "mlops-pipeline-v1" --project "gao-hongnan"
```

```bash
gaohn@<VM_NAME> $ git clone https://github.com/gao-hongnan/common-utils.git
```

```bash
gaohn@<VM_NAME> $ cd common-utils/examples/containerization/docker/mlflow
```

Then we echo something like the below to `.env` file.

```bash
echo -e "# Workspace storage for running jobs (logs, etc)\n\
WORKSPACE_ROOT=/tmp/workspace\n\
WORKSPACE_DOCKER_MOUNT=mlflow_workspace\n\
DB_DOCKER_MOUNT=mlflow_db\n\
\n\
# db\n\
POSTGRES_VERSION=13\n\
POSTGRES_DB=mlflow\n\
POSTGRES_USER=postgres\n\
POSTGRES_PASSWORD=mlflow\n\
POSTGRES_PORT=5432\n\
\n\
# mlflow\n\
MLFLOW_IMAGE=mlflow-docker-example\n\
MLFLOW_TAG=latest\n\
MLFLOW_PORT=5001\n\
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@db:5432/mlflow\n\
MLFLOW_ARTIFACT_STORE_URI=gs://gaohn/imdb/artifacts" > .env
```

Finally, run `bash build.sh` to build the docker image and run the container.

Once successful, you can then access the MLFlow UI at
`http://<EXTERNAL_VM_IP>:5001`.

## Containerization

## Good Practices

- Tagging the image with the commit hash of the code used to build the image.

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


