# Flow

[![Continuous Integration](https://github.com/gao-hongnan/mlops-pipeline-imdb/actions/workflows/continuous_integration.yaml/badge.svg)](https://github.com/gao-hongnan/mlops-pipeline-imdb/actions/workflows/continuous_integration.yaml)

## GOOGLE DIAGRAM

The terminology of "Orchestrated Experiment" and "Automated Pipeline" in Google's MLOps diagram refers to different stages of a machine learning project.

Orchestrated Experiment: This stage typically occurs in a development or experimental setting. Here, data scientists or machine learning engineers are exploring different models, feature engineering strategies, and hyperparameters to find the most effective solution for the problem at hand. The steps in this stage, while methodical, often involve trial and error, exploration, and backtracking. Orchestrating these experiments means organizing and managing them in a systematic way, often using tools such as notebooks (e.g., Jupyter), experiment tracking tools (e.g., MLflow, TensorBoard), version control (e.g., Git), and data versioning tools (e.g., DVC).

Automated Pipeline: Once an effective solution has been found in the experiment stage, the next step is to automate this solution so that it can be run repeatedly and reliably. This involves translating the experimental code into a production-grade pipeline, which includes not only the model training code, but also data extraction, preprocessing, validation, model serving, and monitoring. This pipeline is typically set up to run automatically, either on a regular schedule or in response to certain triggers. The purpose of automation is to ensure consistency, efficiency, and reliability. It allows the machine learning solution to operate at scale and in real-world environments.

Essentially, the orchestrated experiment stage is about finding the best solution, while the automated pipeline stage is about deploying and operating that solution at scale. Both stages involve similar steps (like data validation, data preparation, model training, model evaluation, and model validation), but the context and objectives are different.

## Confusion

I think it dawned upon me after re-reading mlops diagram from google.

They have two components, one is exp/dev environment where ml engineers offline extract data to somewhere for  data analysis. Subsequently, google boxed the follow five steps under "orchestrated experiment".

- data validation
- data preparation
- model training
- model evaluation
- model validation

Then they point this boxed area to another box called source code and point it
to another box called source repository. Subsequently, they point this
source repository to another box called pipeline deployment. The key is now this
pipeline deployment cross over to the second component, namely the staging/production
environment. Let me detail the box in the staging/production environment.

The box is called "automated pipeline" where it contains the following steps:

- data extraction (here i see an arrow from a data warehouse/feature store)
- data validation
- data preparation
- model training
- model evaluation
- model validation

then this pipeline goes to model registry to CD: model serving.

What I did wrong with the dvc is that I included the dvc add and push in the
production environment. This is wrong. The dvc add and push should be done in
the exp/dev environment. The dvc pull should be done in the production environment.
Am i correct? Please dont hesitate to correct me.

---

Let's say I am happy with a particular version of the data i have in my local
dev. Then i commit BOTH THE CODE AND THE DATA TO GIT. Note at this stage, the data and git commit should be in sync. Then i push to git to trigger the CI/CD pipeline.

but now i have issue with pulling the data.

my old extract code

```python
# connection as arg to obey dependency inversion/injection principle
def extract_from_data_warehouse(
    connection: Connection,
    query: str,
    logger: Logger,
    metadata: Metadata,
) -> Metadata:

    logger.info("Starting data extraction...")

    try:
        # assuming that the connection object has a `query` method
        raw_df: pd.DataFrame = connection.query(query)
        logger.info("✅ Data extraction completed. Updating metadata...")

        num_rows, num_cols = raw_df.shape
        dataset, table_name = connection.dataset, connection.table_name

        attr_dict = {
            "raw_df": raw_df,
            "raw_num_rows": num_rows,
            "raw_num_cols": num_cols,
            "raw_dataset": dataset,
            "raw_table_name": table_name,
            "raw_query": query,
        }

        metadata.set_attrs(attr_dict)

        return metadata
    except Exception as error:
        logger.error(f"❌ Data extraction failed. Error: {error}")
        raise error
```

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

## Flow

1. Extract from Data Warehouse where the table from the
    warehouse is considered `raw`, `staging` and `immutable`.
    This table is most likely a result of a typical
    `ETL` or `ELT` process where the data is extracted from
    a source system such as an API, then transformed and
    loaded into a data warehouse as well as a data lake. This
    data is for downstream consumption and is considered
    immutable.
2. Validate the data. This is to ensure that the data has
    expected values and is in the expected format. See
    GCP's CICD blog.
3. Load the data to local and Data Lake. At this stage, if you
    do `tree data` you should see the following:

    ```tree
    .
    ├── raw/
    │   └── <table_name>.csv
    ```

    and of similar structure for the data lake.

4. **Data Versioning Alert**: One super important step is to hash
    the data. Let's see the example.

    - I loaded the data `data.csv` from the data warehouse
        to the local folder `data/raw`.
    - I then hashed the data using `dvc add data/raw/data.csv`.
    - The output is

        ```json
        {
            "filename": "filtered_movies_incremental.csv",
            "filepath": ".cache/483b93622aea897850c63597e852bb25",
            "hash": "483b93622aea897850c63597e852bb25"
        }
        ```
    - I then added the file to the git repo using `git add data/raw/data.csv.dvc`.
    - Now this particular file is tracked by git and dvc.
    - Say the `git` commit hash is `1234567` and the `dvc` commit hash is `483b93622aea897850c63597e852bb25`.
    - When i checkout the `git` commit hash `1234567`, the `dvc` commit hash is still `483b93622aea897850c63597e852bb25`,
    allowing me to checkout the data at that particular commit hash.

## TODOs

1. Remember to compose all the configs like `Metadata` and `Directories` into `Config` for final refactoring.
2. Spend time describing my `Logger` class!
3. Spend time talking about my `DVC` implementation.


## Docker Stuff

tag by commit hash!!

```bash
docker build -t pipeline-training:latest .
docker run -it pipeline-training bash
docker run -it pipeline-training:latest /bin/bash
```

