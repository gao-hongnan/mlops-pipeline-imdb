"""See https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
model validation section."""


def get_production_model_performance(client: MlflowClient, model_name):
    # pull the production models
    production_models = client.get_latest_versions(model_name, stages=["Production"])

    # If there's a production model, compare the new model to it
    # hard assumption that there is only one production model
    production_model = production_models[0]
    production_metrics = production_model.run_data.metrics
    return production_metrics


def compare_models(
    production_metrics, curr_model_metrics, metric_name: str = "test_f1"
):
    # Compare the performance metrics of the two models
    # This is a simple comparison, you might want to implement more robust model comparison logic.
    return curr_model_metrics[metric_name] > production_metrics[metric_name]


def transition_model_to_production(
    client: MlflowClient, model_name: str, model_version: int
):
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
    )
    print(f"Model {model_name} version {model_version} is now in production.")


def find_best_model_for_production(client: MlflowClient, model_name: str = "imdb"):
    # Get all versions of the model
    model_versions = client.search_model_versions(f"name='{model_name}'")

    # Fetch the test_f1 scores for all model versions
    model_versions_metrics = [
        (version.version, client.get_run(version.run_id).data.metrics.get("test_f1"))
        for version in model_versions
    ]
    # tentatively has None values because i renamed the metric a few times.
    model_versions_metrics = [x for x in model_versions_metrics if x[1] is not None]

    # Find the model version with the highest test_f1 score
    pprint(model_versions_metrics)
    best_model_version, best_model_metrics = max(
        model_versions_metrics, key=lambda x: x[1]
    )

    return best_model_version, best_model_metrics


def check_if_there_exists_production_model(
    client: MlflowClient, model_name: str = "imdb"
) -> bool:
    production_models = client.get_latest_versions(model_name, stages=["Production"])
    return len(production_models) > 0


@app.command()
def promote_to_production(model_name: str = "imdb", metric_name: str = "test_f1"):
    """
    Check if the latest model should be promoted to production.
    """
    client = MlflowClient()

    # Check if there are any models in production
    has_production_model: bool = check_if_there_exists_production_model(
        client, model_name
    )

    # no production model yet, promote the best model in the current stage
    if not has_production_model:
        # Find the model version with the highest test_f1 score
        best_model_version, _best_model_metrics = find_best_model_for_production(
            client, model_name
        )
        transition_model_to_production(client, model_name, best_model_version)
        # exit
        return
    else:
        production_metrics = get_production_model_performance(client, model_name)

        # get current model
        non_production_latest_model = client.get_latest_versions(
            model_name, stages=["None"]
        )
        # get current model version and metrics
        curr_model_version = non_production_latest_model[0].version
        curr_model_metrics = non_production_latest_model[0].run_data.metrics

        if not compare_models(
            production_metrics, curr_model_metrics, metric_name=metric_name
        ):
            print("New model did not outperform the current production model.")
            return

        # Promote the best model version to production
        transition_model_to_production(client, model_name, curr_model_version)
