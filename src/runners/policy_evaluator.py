from models.reward_predictor_models import get_reward_predictor_model
from models.estimators import get_estimator_model
from models.nudge_optimizers import get_nudge_optimizer_model
from pydantic import BaseModel
from typing import Dict, List
from dataset.dataset import NudgeOptimizationDataset
from entities.key_entities import NudgeOptimizationDataSpec
from entities.modeling_configs import NudgeOptimizationConfig, RewardPredictorConfig, EstimatorConfig
import pandas as pd
class Runner(BaseModel):
    """Simple runner to run the evaluation setting

    Config Parameters:
        dataset_spec : Used for specifying the columns which correspond to state, action, reward etc.
        reward_predictor : Configuration for creating the reward predictor
        nudge_optimizer : Configuration for creating the nudge optimizer
        estimators : Prints the estimate of net reward for the reward predictions and nudge policy. Multiple
                    estimates can be provided, hence it is a list of estimators.
    """
    dataset_spec: NudgeOptimizationDataSpec
    reward_predictor: RewardPredictorConfig
    nudge_optimizer: NudgeOptimizationConfig
    estimators: List[EstimatorConfig]

    def __call__(self):
        dataset = NudgeOptimizationDataset(self.dataset_spec)
        nudge_optimizer_model = get_nudge_optimizer_model(self.nudge_optimizer.model_class)(self.dataset_spec)
        nudge_optimizer_model.set_params(self.nudge_optimizer.model_parameters)
        estimator_models = [get_estimator_model(estimator.model_class)(self.dataset_spec) for estimator in self.estimators]
        for estimator, estimator_model in zip(self.estimators, estimator_models):
            estimator_model.set_params(estimator.model_parameters)
        
        reward_predictor_model = get_reward_predictor_model(self.reward_predictor.model_class)(self.dataset_spec)
        
        reward_predictor_model.set_params(self.reward_predictor.model_parameters)
        reward_predictor_model(dataset)
        nudge_optimizer_model(dataset)
        for estimator, estimator_model in zip(self.estimators, estimator_models):
            print(estimator.model_class, estimator_model(dataset))

