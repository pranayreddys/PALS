from models.base_models import BaseTabularPredictor
from entities.key_entities import NudgeOptimizationDataSpec
from dataset.dataset import NudgeOptimizationDataset
import numpy as np

def get_estimator_model(model_name):
    """
    This function is called to fetch the appropriate model based on the training config.
    """
    return eval(model_name)


class BaseEstimator(BaseTabularPredictor):
    """
    Assuming actions are converted into numeric values earlier which are 0-indexed.
    Also assuming that predicted reward (computed by models in reward_predictor_models.py) and predicted
    policy columns (computed by models in nudge_optimizers.py) have already been called on the DataSet.

    The estimators then use the reward predictions along with the nudge optimizer to evaluate the average reward
    in the _predict function.
    """
    def __init__(self, _dataspec : NudgeOptimizationDataSpec):
        super().__init__(_dataspec)
        assert isinstance(_dataspec, NudgeOptimizationDataSpec)
    
    def __call__(self, data: NudgeOptimizationDataset):
        possible_actions = self._dataspec.sampling_policy
        self.predicted_policy_cols = ["policy_%s_predicted"%str(action) for action in range(len(possible_actions))]
        self.predicted_reward_cols = ["reward_%s_predicted"%str(action) for action in range(len(possible_actions))]
        return self._predict(
            (
            self._get_data(data.data, self._dataspec.sampling_policy),
            self._get_data(data.data, self.predicted_policy_cols), 
            self._get_data(data.data, self.predicted_reward_cols),
            self._get_data(data.data, self._dataspec.action_column),
            self._get_data(data.data, self._dataspec.reward_column)
            )
        )

class DoublyRobustEstimator(BaseEstimator):
    """Doubly Robust estimator, implemented the equation https://www.cs.cornell.edu/~adith/CfactSIGIR2016/Evaluation2.pdf
    """
    def _predict(self, state):
        sampling_policy, policy_vals, reward_vals, action, reward = state
        action_as_ind = np.array(action, dtype=int)
        recommendation_probability = policy_vals[np.arange(policy_vals.shape[0]),action_as_ind]
        sampling_probability = sampling_policy[np.arange(policy_vals.shape[0]),action_as_ind]
        reward_prediction = reward_vals[np.arange(policy_vals.shape[0]), action_as_ind]
        return np.average((policy_vals*reward_vals).sum(axis=1)) \
                + np.average((recommendation_probability/sampling_probability)*(reward-reward_prediction))

    def set_params(self, params):
        pass

class SelfNormalizedEstimator(BaseEstimator):
    """Self-normalized estimator, implemented the equation https://www.cs.cornell.edu/~adith/CfactSIGIR2016/Evaluation2.pdf
    """
    def _predict(self, state):
        sampling_policy, policy_vals, reward_vals, action, reward = state
        action_as_ind = np.array(action, dtype=int)
        recommendation_probability = policy_vals[np.arange(policy_vals.shape[0]),action_as_ind]
        sampling_probability = sampling_policy[np.arange(policy_vals.shape[0]),action_as_ind]
        normalizing_factor = (recommendation_probability/sampling_probability)
        return np.sum(normalizing_factor * reward)/np.sum(normalizing_factor)

    def set_params(self, params):
        pass

class IPS(BaseEstimator):
    """IPS estimator, implemented the equation https://www.cs.cornell.edu/~adith/CfactSIGIR2016/Evaluation1.pdf
    """
    def _predict(self, state):
        sampling_policy, policy_vals, reward_vals, action, reward = state
        action_as_ind = np.array(action, dtype=int)
        recommendation_probability = policy_vals[np.arange(policy_vals.shape[0]),action_as_ind]
        sampling_probability = sampling_policy[np.arange(policy_vals.shape[0]),action_as_ind]
        normalizing_factor = (recommendation_probability/sampling_probability)
        return np.mean(normalizing_factor * reward)

    def set_params(self, params):
        pass

class ModelTheWorldEstimator(BaseTabularPredictor):
    """
    Model the world estimator that performs naive estimation by simply performing (predicted_reward * policy). If we have some oracle columns,
    then can set those oracle columns in params. The estimator would default to using them (useful if testing on self-generated data)
    """
    def __init__(self, _dataspec : NudgeOptimizationDataSpec):
        super().__init__(_dataspec)
        assert isinstance(_dataspec, NudgeOptimizationDataSpec)
    
    def __call__(self, data: NudgeOptimizationDataset):
        possible_actions = self._dataspec.sampling_policy
        self.predicted_policy_cols = ["policy_%s_predicted"%str(action) for action in range(len(possible_actions))]
        if self.oracle:
            self.predicted_reward_cols = self.oracle_columns
        else:
            self.predicted_reward_cols = ["reward_%s_predicted"%str(action) for action in range(len(possible_actions))]
        return self._predict(
            (
            self._get_data(data.data, self.predicted_policy_cols), 
            self._get_data(data.data, self.predicted_reward_cols),
            )
        )
    
    def _predict(self, data):
        policy, reward = data
        return np.average((policy * reward).sum(axis=1))
    
    def set_params(self, params):
        """Only required if oracle reward columns are present. 

        Args:
            params (Dict): Empty dict can be passed unless oracle predictions are also present.
                            In the case of oracle predictions, oracle_columns & oracle have to be passed in as params. 
        """
        if "oracle" not in params.keys():
            return
        self.oracle = params["oracle"]
        if params["oracle"]:
            self.oracle_columns = params["oracle_columns"]