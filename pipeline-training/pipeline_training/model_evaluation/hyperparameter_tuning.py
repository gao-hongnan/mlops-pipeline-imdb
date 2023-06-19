"""NOTE: As mentioned, this does not touch on
tricks and tips on how to improve the model performance."""
import copy
from typing import Any, Dict, Union

import optuna
from optuna.integration.mlflow import MLflowCallback
from rich.pretty import pprint
from types import SimpleNamespace
from conf.metadata import Metadata

from pipeline_training.model_training.train import train_model


# create an Optuna objective function for hyperparameter tuning
def objective(cfg, logger, metadata, trial: optuna.trial._trial.Trial):
    # define hyperparameters to tune
    # the actual hyperparameters depend on your model
    logger.warning("Performing a deepcopy of the config object to avoid mutation.")
    cfg = copy.deepcopy(cfg)
    cfg.train.vectorizer["analyzer"] = trial.suggest_categorical(
        "vectorizer__analyzer",
        cfg.hyperparameter.hyperparams_grid["vectorizer__analyzer"],
    )
    ngram_range_max = trial.suggest_int(
        "vectorizer__ngram_range_max",
        cfg.hyperparameter.hyperparams_grid["vectorizer__ngram_range"][0],
        cfg.hyperparameter.hyperparams_grid["vectorizer__ngram_range"][1],
    )
    cfg.train.vectorizer["ngram_range"] = (
        cfg.train.vectorizer["ngram_range"][0],
        ngram_range_max,
    )
    cfg.train.model["alpha"] = trial.suggest_loguniform(
        "model__alpha",
        *cfg.hyperparameter.hyperparams_grid["model__alpha"],
    )
    cfg.train.model["power_t"] = trial.suggest_uniform(
        "model__power_t", *cfg.hyperparameter.hyperparams_grid["model__power_t"]
    )

    # you assign back to cfg.train.model and vectorizer so train_model can use them
    # train and evaluate the model using the current hyperparameters
    metadata = train_model(cfg, logger, metadata, trial)
    overall_performance = metadata.model_artifacts["overall_performance"]
    trial.set_user_attr("test_accuracy", overall_performance["test_accuracy"])
    trial.set_user_attr("test_f1", overall_performance["test_f1"])
    trial.set_user_attr("test_loss", overall_performance["test_loss"])
    return overall_performance["test_loss"]


def create_pruner(pruner_config: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_class = eval(pruner_config.pop("pruner_name"))
    pruner = pruner_class(**pruner_config)
    return pruner


def create_sampler(sampler_config: Dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_class = eval(sampler_config.pop("sampler_name"))
    sampler = sampler_class(**sampler_config)
    return sampler


def optimize(cfg, logger, metadata) -> Union[Metadata, SimpleNamespace]:
    logger.info(
        """Seeing inside objective function as well to ensure the hyperparam grid is seeded.
        See https://optuna.readthedocs.io/en/stable/faq.html for how to seed in Optuna"""
    )

    pruner = create_pruner(cfg.hyperparameter.pruner)
    sampler = create_sampler(cfg.hyperparameter.sampler)
    study = optuna.create_study(
        pruner=pruner, sampler=sampler, **cfg.hyperparameter.create_study
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=cfg.exp.tracking_uri, metric_name="test_loss"
    )

    study.optimize(
        lambda trial: objective(cfg, logger, metadata, trial),
        n_trials=cfg.hyperparameter.n_trials,
        callbacks=[mlflow_callback],
    )

    # print the best hyperparameters
    trials_df = study.trials_dataframe()
    pprint(trials_df)
    trials_df = trials_df.sort_values(by=["user_attrs_test_loss"], ascending=False)

    metadata.best_params = {**study.best_trial.params}
    metadata.best_params["best_trial"] = study.best_trial.number

    # update the config object with the best hyperparameters
    # FIXME: here is hardcoded and is prone to error
    cfg.train.vectorizer["analyzer"] = study.best_params["vectorizer__analyzer"]
    cfg.train.vectorizer["ngram_range"] = (
        cfg.train.vectorizer["ngram_range"][0],
        study.best_params["vectorizer__ngram_range_max"],
    )
    cfg.train.model["alpha"] = study.best_params["model__alpha"]
    cfg.train.model["power_t"] = study.best_params["model__power_t"]
    return metadata, cfg
