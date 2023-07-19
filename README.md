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


