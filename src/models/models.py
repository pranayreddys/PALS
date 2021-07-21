from models.base_models import BaseTimeSeriesModel
from entities.modeling_configs import TimeSeriesTrainingConfig
from entities.key_entities import TimeSeriesDataSpec
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def get_model(ts_config: TimeSeriesTrainingConfig):
    return eval(ts_config.model_class)

class SimpleVAR(BaseTimeSeriesModel):    
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)
        self.linear = layers.Dense(len(_dataspec.independent_state_columns)+len(_dataspec.dependent_state_columns))


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        state_vals = tf.reshape(state_vals,[state_vals.shape[0], -1])
        # control_input_vals.view((control_input_vals.shape[0], -1))
        out = self.linear(state_vals)
        return out
    
    def set_params(self, params):
        pass
