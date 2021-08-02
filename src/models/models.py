from models.base_models import BaseTimeSeriesModel
from entities.modeling_configs import TimeSeriesTrainingConfig
from entities.key_entities import TimeSeriesDataSpec
from tensorflow.keras import layers, regularizers
import tensorflow as tf
import numpy as np

def get_model(ts_config: TimeSeriesTrainingConfig):
    """
    This function is called to fetch the appropriate model based on the training config.
    """
    return eval(ts_config.model_class)

"""
All the classes below inherit from the BaseTimeSeriesModel class, implementing the 
_predict_one_step() function in different manners.

"""
class SimpleVAR(BaseTimeSeriesModel):    
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        state_vals = tf.reshape(state_vals,[state_vals.shape[0], -1])
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        out = self.linear(tf.concat([state_vals, control_input_vals], axis=1))
        return out
    
    def set_params(self, params):
        self.linear = layers.Dense(
            len(self.dataspec.independent_state_columns)+len(self.dataspec.dependent_state_columns)
        )


class UatVAR(BaseTimeSeriesModel):    
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        out = []
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        for i in range(len(self.dataspec.independent_state_columns)):
            out.append(self.linears[i](tf.concat([tf.reshape(state_vals[:,:, i], [state_vals.shape[0], -1]), control_input_vals], axis=1)))       
        independent_state_predictions = tf.concat(out, axis=1)
        return independent_state_predictions
    
    def set_params(self, params):
        self.linears = []
        for _ in range(len(self.dataspec.independent_state_columns)):
            self.linears.append(layers.Dense(1))

class UatBpVAR(BaseTimeSeriesModel):    
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        out = []
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        num_independent = len(self.dataspec.independent_state_columns)
        for i in range(num_independent):
            out.append(self.linears[i](tf.concat([tf.reshape(state_vals[:,:, i], [state_vals.shape[0], -1]), control_input_vals], axis=1)))

        independent_state_predictions = tf.concat(out, axis=1)
        out = []
        for i in range(num_independent, num_independent+len(self.dataspec.dependent_state_columns)):
            out.append(self.linears[i](tf.concat([tf.reshape(state_vals[:,:, i], [state_vals.shape[0], -1]), independent_state_predictions], axis=1)))       
        
        dependent_state_predictions = tf.concat(out, axis=1)
        return tf.concat([independent_state_predictions, dependent_state_predictions], axis=1)
    
    def set_params(self, params):
        self.linears = []
        for _ in range(len(self.dataspec.independent_state_columns)+len(self.dataspec.dependent_state_columns)):
            self.linears.append(layers.Dense(1))


class DelayedEffectModel(BaseTimeSeriesModel):
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        out = []
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        num_independent = len(self.dataspec.independent_state_columns)
        for i in range(num_independent):
            out.append(state_vals[:,-1,i]
                + self.linears[i](tf.concat([tf.reshape(state_vals[:,:, i], [state_vals.shape[0], -1]), control_input_vals], axis=1)))

        independent_state_predictions = tf.concat(out, axis=1)
        out = []
        for i in range(num_independent, num_independent+len(self.dataspec.dependent_state_columns)):
            out.append(state_vals[:,-1,i]
                + self.linears[i](tf.concat([tf.reshape(state_vals[:,:, i], [state_vals.shape[0], -1]), independent_state_predictions], axis=1)))       
        
        dependent_state_predictions = tf.concat(out, axis=1)
        return tf.concat([independent_state_predictions, dependent_state_predictions], axis=1)
    
    def set_params(self, params):
        self.linears = []
        for _ in range(len(self.dataspec.independent_state_columns)+len(self.dataspec.dependent_state_columns)):
            self.linears.append(layers.Dense(1))
