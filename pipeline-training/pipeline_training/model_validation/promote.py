"""See https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
model validation section.

This script handles promoting a model to production.
Simple logic is if the new model is better than the current production model, then promote it to production.
Of course in real life, you might want to implement more robust model comparison logic and
implement a manual approval step before promoting the model to production. Include A/B testing too."""

from rich.pretty import pprint
from common_utils.core.logger import Logger


from typing import Dict, List, Optional, Union, Literal
from rich.pretty import pprint

from conf.metadata import Metadata
from dataclasses import dataclass
from typing import Dict
from types import SimpleNamespace

from abc import ABC, abstractmethod


@dataclass
class ProductionModel:
    """
    A dataclass model representing a version of a machine learning model.
    The use of this class adheres to the Single Responsibility Principle as it only has the responsibility of
    encapsulating the logic related to a model version.
    Dataclass enforces type checking at runtime and provides helpful error messages when data is invalid.

    Value Object Pattern.
    """

    version: int
    metrics: Dict[str, float]


class ModelClient(ABC):
    """
    Abstract base class (interface) for a model client.
    """

    @abstractmethod
    def get_latest_versions(
        self, model_name: str, stages: List[str]
    ) -> List[ProductionModel]:
        pass

    # Define other necessary methods...


class MlflowClient(ModelClient):
    """
    Concrete class for Mlflow client.
    """

    def get_latest_versions(
        self, model_name: str, stages: List[str]
    ) -> List[ProductionModel]:
        # Implement the method for Mlflow
        pass

    # Implement other necessary methods...


class WandbClient(ModelClient):
    """
    Concrete class for Wandb client.
    """

    def get_latest_versions(
        self, model_name: str, stages: List[str]
    ) -> List[ProductionModel]:
        # Implement the method for Wandb
        pass

    # Implement other necessary methods...


class ModelPromotionManager:
    """A class that manages the promotion of machine learning models to production.

    This class adheres to several principles of good software design:

    1. Single Responsibility Principle (SRP): This class's responsibility is clear -
       managing the promotion of models to production. It maintains the current and
       previous versions of a model, and decides whether to promote a new model
       based on a comparison of their performances.

    2. Open-Closed Principle (OCP): This class is open for extension (e.g., one can
       easily extend it to use different promotion strategies) but closed for
       modification (adding new functionality doesn't require modification of the
       existing code).

    3. Liskov Substitution Principle (LSP): This principle is not explicitly
       implemented in this class since it doesn't have any subclasses. However, if a
       subclass were created (e.g., a `ThresholdPromotionManager` that promotes
       models that exceed a certain performance threshold), it could be substituted
       for this class without altering any of the properties that this class
       guarantees (e.g., that a model won't be demoted).

    4. Interface Segregation Principle (ISP): The `ModelPromotionManager` class does
       not depend on any interfaces it doesn't use. The class itself can be seen as
       a high-level interface for model promotion operations.

    5. Dependency Inversion Principle (DIP): The `ModelPromotionManager` class
       depends on abstractions (e.g., the `MlflowClient` class and the
       `ProductionModel` class), not on concrete classes or implementations.

    NOTE: For design pattern consider using the Strategy Pattern.
    To allow for various types of clients like Mlflow, Wandb, etc., you could employ
    the Strategy Design Pattern. This pattern enables a strategy (algorithm) to be
    selected at runtime. It is one of the behavioral design patterns and it's quite
    useful when you have a set of similar algorithms and want to switch between them
    dynamically.

    In the context of your ModelPromotionManager class, the "strategy" would be the
    different types of clients you want to support (like Mlflow, Wandb, etc.). You
    could define an interface that each client must implement. Then, in your
    ModelPromotionManager class, you would use this interface instead of a specific
    client implementation.

    This change would make your ModelPromotionManager class adhere more closely to
    the Open/Closed Principle (part of the SOLID principles), because it would be
    open for extension (you can add support for new types of clients) but closed for
    modification (you don't have to change the ModelPromotionManager class to add
    new clients).
    """

    def __init__(
        self,
        client: ModelClient,
        logger: Logger,
        metadata: Metadata,
        cfg: SimpleNamespace,
    ):
        self.client = client
        self.logger = logger
        self.metadata = metadata
        self.cfg = cfg

        self.model_name = cfg.exp.register_model["name"]

    def _check_if_there_exists_production_model(self) -> Literal[True, False]:
        return len(self._get_all_production_models()) > 0

    def _get_all_production_models(self) -> List:
        production_models = self.client.get_latest_versions(
            self.model_name, stages=["Production"]
        )
        return production_models

    def _get_latest_production_model(self) -> ProductionModel:
        """
        Fetch the current production model, if one exists.
        Assume latest production model is the one with the highest version number
        and also the one deployed most recently.
        """
        production_models = self._get_all_production_models()
        sorted_production_models = sorted(
            production_models, key=lambda model: model.version, reverse=True
        )

        # get the latest production model
        latest_production_model = sorted_production_models[0]
        run_id = latest_production_model.run_id
        run = self.client.get_run(run_id)
        return ProductionModel(
            version=latest_production_model.version,
            metrics=run.data.metrics,
        )

    def _find_best_model_for_production(
        self, metric_name: str = "test_f1"
    ) -> ProductionModel:
        """
        Find the best model to promote to production based on the metric score.
        This is needed for the first time when there is no production models.
        """
        model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        model_versions_metrics = [
            (
                version.version,
                self.client.get_run(version.run_id).data.metrics.get(metric_name),
            )
            for version in model_versions
        ]
        model_versions_metrics = [x for x in model_versions_metrics if x[1] is not None]
        best_model_version, best_model_metrics = max(
            model_versions_metrics, key=lambda x: x[1]
        )
        return ProductionModel(best_model_version, best_model_metrics)

    def _compare_models(
        self,
        model1: ProductionModel,
        model2: ProductionModel,
        metric_name: str = "test_f1",
    ) -> bool:
        """
        Compare two models based on their performance metric.

        Args:
            model1: The first model to compare.
            model2: The second model to compare.
            metric_name: The name of the performance metric to use for the comparison.

        Returns:
            bool: True if model1's performance is better than model2's performance, False otherwise.
        """
        return model1.metrics.get(metric_name, 0) > model2.metrics.get(metric_name, 0)

    def _transition_model_to_production(self, model_version: int) -> None:
        """
        Transition a model to the production stage.

        Args:
            model_version: The version number of the model to transition to production.
        """
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version,
            stage="Production",
        )
        self.logger.info(
            f"Model {self.model_name} version {model_version} is now in production."
        )

    def promote_to_production(self, metric_name: str = "test_f1") -> None:
        """
        Promote the best model version to production if it outperforms the current production model.

        Args:
            metric_name: The name of the performance metric to use for the comparison.
        """
        has_production_model: bool = self._check_if_there_exists_production_model()

        # no production model yet, promote the best model in the current stage
        if not has_production_model:
            # Find the model version with the highest test_f1 score
            curr_best_model: ProductionModel = self._find_best_model_for_production(
                metric_name=metric_name
            )
            pprint(curr_best_model)
            self._transition_model_to_production(curr_best_model.version)
            # exit
            return
        else:
            latest_production_model = self._get_latest_production_model()

            # get current model, note we do not need to get all the model versions
            # because we assume the latest model version is the one we want to promote
            non_production_latest_model = self.client.get_latest_versions(
                self.model_name, stages=["None"]
            )[0]

            # get the model version number and metrics
            latest_curr_model_version = non_production_latest_model.version
            latest_curr_model_run_id = non_production_latest_model.run_id
            latest_curr_model_run = self.client.get_run(latest_curr_model_run_id)

            latest_curr_model_metrics = latest_curr_model_run.data.metrics
            latest_curr_model: ProductionModel = ProductionModel(
                version=latest_curr_model_version, metrics=latest_curr_model_metrics
            )

            if not self._compare_models(
                model1=latest_curr_model, model2=latest_production_model
            ):
                self.logger.info(
                    f"Model {self.model_name} version {latest_curr_model_version} does not outperform the current production model."
                )
                return
            self._transition_model_to_production(latest_curr_model_version)
