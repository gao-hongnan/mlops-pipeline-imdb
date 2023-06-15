import mlflow

# from common_utils.core.logger import Logger
# from rich.pretty import pprint

# from conf.init_dirs import ROOT_DIR
# from conf.init_project import initialize_project

# cfg = initialize_project(ROOT_DIR)

tracking_uri = "http://34.94.42.137:5001"
experiment_name = "imdb_revamp"
run_name = "untuned_imdb_sgd_dummy"
nested = False

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name=experiment_name)


def train():
    # nested=True because this is nested under a parent train func in main.py.
    with mlflow.start_run(run_name=run_name, nested=nested):
        run_id = mlflow.active_run().info.run_id
        print(f"run_id: {run_id}")
        # mlflow.log_artifacts(
        #     "/Users/gaohn/gaohn/end2end-movie-recommender-system/pipeline-training/conf"
        # )
