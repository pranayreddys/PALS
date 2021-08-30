"""The code here is utilized for reward prediction modules for the counterfactual evaluation.
"""
from entities.key_entities import NudgeOptimizationDataSpec
import numpy as np
from typing import Dict
from models.base_models import BaseTabularPredictor
from models.models import get_forecasting_model
from dataset.dataset import NudgeOptimizationDataset, TimeSeriesDataset
import tensorflow as tf
import pandas as pd
from pydantic import parse_file_as
from runners.model_runner import Runner

def get_reward_predictor_model(model_name):
    """
    This function is called to fetch the appropriate model based on the training config.
    Note that any new model implemented below also needs to be added in all_enums.py in the RewardPredictorClass enum.
    """
    return eval(model_name)


class BaseRewardPredictor(BaseTabularPredictor):
    """Base class for reward predictors
    """
    def __init__(self, _dataspec: NudgeOptimizationDataSpec):
        super().__init__(_dataspec)
        assert isinstance(_dataspec, NudgeOptimizationDataSpec)

    def __call__(self, data: NudgeOptimizationDataset):
        """

        Args:
            data (NudgeOptimizationDataset): [description]

        Returns:
            [NudgeOptimizationDataset]: Returns reward for each possible action by adding additional columns to the
            NudgeOptimizationDataset passed. The additional columns are named reward_'actionnum'_predicted.
            The _predict function needs to be overriden by child classes
        """
        possible_actions = self._dataspec.sampling_policy
        predicted_reward = ["reward_%s_predicted"%str(action) for action in range(len(possible_actions))]
        data.data[predicted_reward] = self._predict(
            self._get_data(data.data, self._dataspec.state_columns)
        )
        return data


class RewardPredictorToy(BaseRewardPredictor):
    """
    Used ONLY in a very simple contextual bandit scenario - meant for testing the estimators.
    This reward predictor is useful as a toy evaluation that estimates the effect of bias and variance
    in reward predictions. Input is assumed to be the actual reward for all possible actions
    """
    def _predict(self, data: np.array):
        self.bias_properties = np.random.uniform(-self.bias, self.bias, (data.shape[1]))
        data += (self.bias_properties + self.noise * np.random.standard_normal(data.shape))
        return data
        
    def set_params(self, params):
        self.noise = params["noise"]
        self.bias = params["bias"]

class RewardPredictorFromTimeSeries(BaseRewardPredictor):
    """
    Expects column naming convention of NudgeOptimizationDataSpec state columns to be similar
    to the column naming convention of TimeSeriesDataSpec, but with an _ character followed by 
    day number at the end. 
    Example: if a feature in the time series data is a column named stepcount, then this reward predictor
    assumes the existence of columns named 'stepcount_0, stepcount_1, ... stepcount_(context_window-1)'.
    TODO: Provide an interface for changing the reward function. Current reward function is simply BP[t+5] - BP[t],
    there are several hacks involved in this predictor, and it needs to be modified.
    TODO: This model is quite slow (takes a few minutes to run), because it is not optimized properly. A better interface
    can be provided at the TimeSeries Model to facilitate this.
    Also look at set_params for initialization of this reward predictor. 
    """
    def convert_to_timeseries(self,df): 
        """Helper function for the nudge optimizer, converts a dataframe into a TimeSeriesDataSet Object

        Args:
            df (pd.Series): Input is a single datapoint consisting of only states for a given time. This state
            is unwrapped into a context_window, lead_gap, forecast_horizon framework 

        Returns:
            [TimeSeriesDataset]: The data point is converted from the NudgeOptimizationDataSpec to the TimeSeriesDataSpec format
            This datapoint will be passed into the TimeSeries model for prediction, and the obtained prediction can then be analyzed
        """
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
            #HACK: To initialize the model, keras requires shape of input, hence there is a call and break here
            self.model.load_model(self.params["model_save_folder"], self.params["prediction_config"], t)
            break
        
        rewards =  []
        for idx, series in nudge_optimizer_data.iterrows():
            ts = self.convert_to_timeseries(series)
            reward_row =  []
            for nudge in range(len(self._dataspec.sampling_policy)):
                #HACK: Nudge indexing used in time series modelling was 1-based (0 was no nudge). In reward prediction, however, we are using 0 based
                #indexing. Thus nudge+1 is input into the model.
                ts.data.loc[ts.data[self.model.dataspec.time_column]>=self.params["prediction_config"].context_window,self._dataspec.action_column] = nudge + 1
                
                
                predictions = self.model.simple_predict(ts, self.params["prediction_config"])

                #HACK hardcoded input here, needs to be modified to be more general. This line needs to be fixed based on use-case, and clinician
                # driven inputs.
                reward = (predictions.data.iloc[0,predictions.data.columns.get_loc("VariableName.sbp_horizon_5_predict")]
                        - predictions.data.iloc[self.params["prediction_config"].context_window-1,predictions.data.columns.get_loc("VariableName.sbp")])
                reward_row.append(reward)

            rewards.append(reward_row)
        return np.array(rewards)
        
    def set_params(self, params):
        """Initialization of the Reward Predictor

        Args:
            params ([Dict]): Consists of the parameters to initialize this model.
            Expects the following keys to be present:
                train_file_path: The runner config utilized for time series model training
                prediction_file_path: The config utilized for prediction.
                Note that although the interface seems very general, this has just been provided for ease
                of use, and the interface is being parsed for picking the actual parameters that we
                are interested in.

                Ideally, this model requires the following keys (which are initialized below)
                training_config ([TimeSeriesTrainingConfig]): Configuration used to train the time series model
                prediction_config ([TimeSeriesPredictionConfig]): Configuration to be used for prediction in the time series model
                model_save_folder ([str]): Folder in which the trained time series model's weights have been saved
                dataspec ([TimeSeriesDataSpec]): The dataspec of the time series model (which has the independent,dependent state variables etc)
                
                Unless lead gap/ forecast horizon is being changed during inference, it is alright to keep the training and prediction configs
                same.
        """
        self.params = params
        runner_config = parse_file_as(Runner, params["train_file_path"])
        params["training_config"] = runner_config.training_config
        params["prediction_config"] = runner_config.training_config
        params["dataspec"] = runner_config.dataset_spec
        params["model_save_folder"] = runner_config.training_config.model_save_folder
        self.model = get_forecasting_model(params["prediction_config"].model_class)(params["dataspec"])
        self.model.set_params(params["training_config"].model_parameters)