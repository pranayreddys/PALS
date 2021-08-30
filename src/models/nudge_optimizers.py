import abc
from entities.key_entities import NudgeOptimizationDataSpec
from dataset.dataset import NudgeOptimizationDataset
import pandas as pd
import numpy as np
from typing import Dict
from models.base_models import BaseTabularPredictor
from dataset.dataset import TimeSeriesDataset
from pydantic import parse_file_as
from runners.model_runner import Runner
from models.models import get_forecasting_model

def get_nudge_optimizer_model(model_name):
    """
    This function is called to fetch the appropriate model based on the training config.
    """
    return eval(model_name)

class BaseNudgeOptimizer(BaseTabularPredictor):
    def __init__(self, _dataspec):
        super().__init__(_dataspec)
        assert isinstance(_dataspec, NudgeOptimizationDataSpec)

    def __call__(self, data: NudgeOptimizationDataset):
        """
        Asssume data contains state columns, and attempts to propose a policy based on the state.

        Args:
            data (NudgeOptimizationDataset): The nudge optimization dataset.

        Returns:
            NudgeOptimizationDataset: Populated NudgeOptimizationDataset with num_actions columns extra.
                                      These columns denote the probability distribution pi(a|s) recommended by the considered policy.
                                      For deterministic policies, for a state s0, only one of pi(a|s0) is 1, while rest are
                                      all 0.
        """
        possible_actions = self._dataspec.sampling_policy # The number of sampling policy columns is same as number of actions
        predicted_policy = ["policy_%s_predicted"%str(action) for action in range(len(possible_actions))]        
        data.data[predicted_policy] = self._predict(
            self._get_data(data.data, self._dataspec.state_columns)
        )
        return data

class NudgeOptimizerToy(BaseNudgeOptimizer):
    """
    Used ONLY in a very simple contextual bandit scenario - meant for testing the estimators
    Do not use in real code. 
    The scenario where this was used when state itself contained the predicted rewards, and the policy
    was simply picking the best action with probability p, and the rest with probability (1-p)/(num_actions-1)
    """
    def _predict(self, state):
        out = np.full((len(state), len(self._dataspec.sampling_policy)), (1-self.p)/(len(self._dataspec.sampling_policy)-1))
        out[np.arange(len(state)),state.argmax(axis=1)] = self.p
        return out
    
    def set_params(self, params):
        self.p = params["p"]

class UniformActionRecommender(BaseNudgeOptimizer):
    """
    Uniform random action, assigned with a probability of 1/num_actions.
    """
    def _predict(self, state):
        out = np.full((len(state), len(self._dataspec.sampling_policy)), 1.0/len(self._dataspec.sampling_policy))
        return out
    
    def set_params(self, params):
        pass

class NudgeOptimizerFromTimeSeries(BaseNudgeOptimizer):
    """
    Almost identical to RewardPredictorFromTimeSeries in reward_predictor_models.py.
    Please refer to the comments there before coming to this class.
    """
    def convert_to_timeseries(self,df): # df consists of only single point
        ret_dataset = TimeSeriesDataset(self.model.dataspec, blank_dataset=True)
        ret_df = pd.DataFrame()
        prediction_config = self.params["prediction_config"]
        total_window_size = (prediction_config.context_window 
                            + prediction_config.lead_gap
                            + prediction_config.forecast_horizon)
        ret_df[self.model.dataspec.time_column] = list(range(total_window_size))
        ret_df[self.model.dataspec.series_id_column] = 1 #HACK since current time series models 
        # do not use series_id_column, just adding a dummy input
        columns = (self.model.dataspec.dependent_state_columns + self.model.dataspec.independent_state_columns + self.model.dataspec.series_attribute_columns) 

        d = {column: [] for column in columns}
        d["nudge"] = [] #HACK: althought time series model has a general control input, assuming that there is a single control input
        # called nudge and changing/passing this control input into the 
        for time in range(prediction_config.context_window):
            for column in columns:
                d[column].append(df[column+"_"+str(time)])
            
            d["nudge"].append(0)
        
        for time in range(prediction_config.lead_gap + prediction_config.forecast_horizon):
            for column in columns:
                d[column].append(0)
            d["nudge"].append(0)
        
        for column, item in d.items():
            ret_df[column] = item
        ret_dataset.data = ret_df
        return ret_dataset

    def _predict(self, data: np.array):
        """
        This function could be optimized further.
        Currently performing a very naive interconversion from nudge dataset to time series dataset.
        There are also many hacks employed here, it is not a general function as it assumes that nudge is only
        a single column in the dataset.
        """
        nudge_optimizer_data= pd.DataFrame(data=data, columns=self._dataspec.state_columns)
        for idx, series in nudge_optimizer_data.iterrows():
            t = self.convert_to_timeseries(series)
            self.model.load_model(self.params["model_save_folder"], self.params["prediction_config"], t)
            break
        
        policy = np.zeros((len(nudge_optimizer_data), len(self._dataspec.sampling_policy)))
        i = 0
        for idx, series in nudge_optimizer_data.iterrows():
            ts = self.convert_to_timeseries(series)
            reward_row =  []
            for nudge in range(len(self._dataspec.sampling_policy)):
                ts.data.loc[ts.data[self.model.dataspec.time_column]>=self.params["prediction_config"].context_window,"nudge"] = nudge + 1
                predictions = self.model.simple_predict(ts, self.params["prediction_config"])
                reward = (predictions.data.iloc[0,predictions.data.columns.get_loc("VariableName.sbp_horizon_5_predict")]
                        - predictions.data.iloc[self.params["prediction_config"].context_window-1,predictions.data.columns.get_loc("VariableName.sbp")])
                reward_row.append(reward)
            policy[i,reward_row.index(min(reward_row))] = 1 
            i += 1
        return policy

    def set_params(self, params):
        self.params = params
        runner_config = parse_file_as(Runner, params["train_file_path"])
        params["training_config"] = runner_config.training_config
        params["prediction_config"] = runner_config.training_config
        params["dataspec"] = runner_config.dataset_spec
        params["model_save_folder"] = runner_config.training_config.model_save_folder
        self.model = get_forecasting_model(params["prediction_config"].model_class)(params["dataspec"])
        self.model.set_params(params["training_config"].model_parameters)