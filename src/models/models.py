from models.base_models import BaseTimeSeriesModel
from entities.modeling_configs import TimeSeriesTrainingConfig
from entities.key_entities import TimeSeriesDataSpec
from entities.all_enums import ModelClass
from tensorflow.keras import layers, regularizers, activations
import tensorflow as tf
import numpy as np
import os
import matplotlib

def get_forecasting_model(model_class: ModelClass):
    """
    This function is called to fetch the appropriate model based on the training config.
    Note that whenever a new model is created, it also needs to be updated in the ModelClass enum
    of src.entitites.all_enums.py 
    """
    return eval(model_class)

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
    
    def visualize(self, folder_path):
        raise NotImplementedError

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
    
    def visualize(self, folder_path):
        raise NotImplementedError

class UatDelayedEffect(BaseTimeSeriesModel):    
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        out = []
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        for i in range(len(self.dataspec.independent_state_columns)):
            out.append(state_vals[:,:, -1]+self.linears[i](tf.concat([tf.reshape(state_vals[:,:, i], [state_vals.shape[0], -1]), control_input_vals], axis=1)))       
        independent_state_predictions = tf.concat(out, axis=1)
        return independent_state_predictions
    
    def set_params(self, params):
        self.linears = []
        for _ in range(len(self.dataspec.independent_state_columns)):
            self.linears.append(layers.Dense(1))
    
    def visualize(self, folder_path: str):
        raise NotImplementedError

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
    
    def visualize(self, folder_path: str):
        raise NotImplementedError

class DelayedEffectModel(BaseTimeSeriesModel):
    """
    Used when there are only control and independent state columns, no dependent state columns
    """
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        return state_vals[:,-1,:]+ self.linear(control_input_vals)
    
    def _custom_l2_regularizer(self,weights):
        return tf.square(tf.reduce_sum(0.05 * weights))

    def set_params(self, params):
        self.num_independent = len(self.dataspec.independent_state_columns)
        self.linear = layers.Dense(self.num_independent)
    
    def visualize(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        
        weights, bias = self.linear.get_weights()
        indep_layer_weights = np.zeros((weights.shape[0]+1, self.num_independent))
        indep_layer_weights[:weights.shape[0]] = weights
        indep_layer_weights[weights.shape[0]:] = bias
        save_path = os.path.join(folder_path, "indep_layer_weights.txt")
        np.savetxt(save_path, indep_layer_weights)
        weights = weights[::-1]
        num_control = len(self.dataspec.control_input_columns)
        for i in range(num_control):
            for j in range(self.num_independent):
                save_path = os.path.join(folder_path, self.dataspec.control_input_columns[num_control-i-1] + "_"
                                                    + self.dataspec.independent_state_columns[j]+"_profile.png")
                matplotlib.pyplot.figure()
                profile = np.cumsum(weights[i::num_control, j])
                matplotlib.pyplot.plot(profile)
                matplotlib.pyplot.xlabel('Time')
                matplotlib.pyplot.ylabel('Effect')
                matplotlib.pyplot.savefig(save_path)
                save_path = os.path.join(folder_path, self.dataspec.control_input_columns[num_control-i-1] + "_"
                                                    + self.dataspec.independent_state_columns[j]+"_profile.txt")
                np.savetxt(save_path, profile)

class BpDelayedEffectModel(BaseTimeSeriesModel):
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        independent_state_vals = (state_vals[:,-1,:self.num_independent]
                                + self.linear_independent(control_input_vals))

        independent_state_inputs = tf.concat([state_vals[:,:, :self.num_independent], tf.reshape(independent_state_vals, [state_vals.shape[0],1,-1])], axis=1)
        dependent_state_vals = state_vals[:,-1,self.num_independent:]+ self.linear_dependent(tf.reshape(independent_state_inputs, [state_vals.shape[0],-1]))
        
        return tf.concat([independent_state_vals, dependent_state_vals], axis=1)
    
    def _custom_l2_regularizer(self,weights):
        return tf.square(tf.reduce_sum(0.05 * weights))
    
    def visualize(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        
        weights, bias = self.linear_independent.get_weights()
        indep_layer_weights = np.zeros((weights.shape[0]+1, self.num_independent))
        indep_layer_weights[:weights.shape[0]] = weights
        indep_layer_weights[weights.shape[0]:] = bias
        save_path = os.path.join(folder_path, "indep_layer_weights.txt")
        np.savetxt(save_path, indep_layer_weights)
        weights = weights[::-1]
        num_control = len(self.dataspec.control_input_columns)
        for i in range(num_control):
            for j in range(self.num_independent):
                save_path = os.path.join(folder_path, self.dataspec.control_input_columns[num_control-i-1] + "_"
                                                    + self.dataspec.independent_state_columns[j]+"_profile.png")
                matplotlib.pyplot.figure()
                profile = np.cumsum(weights[i::num_control, j])
                matplotlib.pyplot.plot(profile)
                matplotlib.pyplot.xlabel('Time')
                matplotlib.pyplot.ylabel('Effect')
                matplotlib.pyplot.savefig(save_path)
                save_path = os.path.join(folder_path, self.dataspec.control_input_columns[num_control-i-1] + "_"
                                                    + self.dataspec.independent_state_columns[j]+"_profile.txt")
                np.savetxt(save_path, profile)
        
        weights, bias = self.linear_dependent.get_weights()
        dep_layer_weights = np.concatenate([weights, bias.reshape(1,-1)], axis=0)        
        save_path = os.path.join(folder_path, "dep_layer_weights.txt")
        np.savetxt(save_path, dep_layer_weights)
        weights = weights[::-1]
        for i in range(self.num_independent):
            for j in range(self.num_dependent):
                save_path = os.path.join(folder_path, self.dataspec.independent_state_columns[self.num_independent-i-1] + "_"
                                                    + self.dataspec.dependent_state_columns[j]+"_profile.png")
                matplotlib.pyplot.figure()
                profile= np.cumsum(weights[i::self.num_independent, j])
                matplotlib.pyplot.plot(profile)
                matplotlib.pyplot.xlabel('Time')
                matplotlib.pyplot.ylabel('Effect')
                matplotlib.pyplot.savefig(save_path)
                save_path = os.path.join(folder_path, self.dataspec.independent_state_columns[self.num_independent-i-1] + "_"
                                                    + self.dataspec.dependent_state_columns[j]+"_profile.txt")
                np.savetxt(save_path, profile)
    
    def _custom_l2_regularizer(self,weights):
        return tf.square(tf.reduce_sum(0.05 * weights))

    def set_params(self, params):
        self.num_independent = len(self.dataspec.independent_state_columns)
        self.num_dependent = len(self.dataspec.dependent_state_columns)
        self.linear_independent = layers.Dense(self.num_independent)
        self.linear_dependent = layers.Dense(self.num_dependent)
    

class BpDelayedEffectModelUserFeature(BaseTimeSeriesModel):
    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super().__init__(_dataspec)


    def _predict_one_step(self, state_vals, control_input_vals):
        # B x T x S
        category = control_input_vals[0, 0, -1]
        control_input_vals = control_input_vals[:,:, :-1]
        control_input_vals = tf.reshape(control_input_vals, [control_input_vals.shape[0], -1])
        independent_state_vals = (state_vals[:,-1,:self.num_independent]
                                + self.linear_independents[int(category)](control_input_vals))

        independent_state_inputs = tf.concat([state_vals[:,:, :self.num_independent], tf.reshape(independent_state_vals, [state_vals.shape[0],1,-1])], axis=1)
        dependent_state_vals = state_vals[:,-1,self.num_independent:]+ self.linear_dependent(tf.reshape(independent_state_inputs, [state_vals.shape[0],-1]))
        
        return tf.concat([independent_state_vals, dependent_state_vals], axis=1)
    
    def _custom_l2_regularizer(self,weights):
        return tf.square(tf.reduce_sum(0.05 * weights))
    
    def visualize(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        
        for person_category, linear_independent in enumerate(self.linear_independents):
            weights, bias = linear_independent.get_weights()
            indep_layer_weights = np.zeros((weights.shape[0]+1, self.num_independent))
            indep_layer_weights[:weights.shape[0]] = weights
            indep_layer_weights[weights.shape[0]:] = bias
            save_path = os.path.join(folder_path, "Category_"+str(person_category)+"_indep_layer_weights.txt")
            np.savetxt(save_path, indep_layer_weights)
            weights = weights[::-1]
            num_control = len(self.dataspec.control_input_columns)
            for i in range(num_control):
                for j in range(self.num_independent):
                    save_path = os.path.join(folder_path, "Category_"+str(person_category)+ "_"+self.dataspec.control_input_columns[num_control-i-1] + "_"
                                                        + self.dataspec.independent_state_columns[j]+"_profile.png")
                    matplotlib.pyplot.figure()
                    profile = np.cumsum(weights[i::num_control, j])
                    matplotlib.pyplot.plot(profile)
                    matplotlib.pyplot.xlabel('Time')
                    matplotlib.pyplot.ylabel('Effect')
                    matplotlib.pyplot.savefig(save_path)
                    save_path = os.path.join(folder_path, "Category_"+str(person_category)+"_"+self.dataspec.control_input_columns[num_control-i-1] + "_"
                                                        + self.dataspec.independent_state_columns[j]+"_profile.txt")
                    np.savetxt(save_path, profile)
        
        weights, bias = self.linear_dependent.get_weights()
        dep_layer_weights = np.concatenate([weights, bias.reshape(1,-1)], axis=0)        
        save_path = os.path.join(folder_path, "dep_layer_weights.txt")
        np.savetxt(save_path, dep_layer_weights)
        weights = weights[::-1]
        for i in range(self.num_independent):
            for j in range(self.num_dependent):
                save_path = os.path.join(folder_path, self.dataspec.independent_state_columns[self.num_independent-i-1] + "_"
                                                    + self.dataspec.dependent_state_columns[j]+"_profile.png")
                matplotlib.pyplot.figure()
                profile= np.cumsum(weights[i::self.num_independent, j])
                matplotlib.pyplot.plot(profile)
                matplotlib.pyplot.xlabel('Time')
                matplotlib.pyplot.ylabel('Effect')
                matplotlib.pyplot.savefig(save_path)
                save_path = os.path.join(folder_path, self.dataspec.independent_state_columns[self.num_independent-i-1] + "_"
                                                    + self.dataspec.dependent_state_columns[j]+"_profile.txt")
                np.savetxt(save_path, profile)
    
    def _custom_l2_regularizer(self,weights):
        return tf.square(tf.reduce_sum(0.05 * weights))

    def set_params(self, params):
        self.num_independent = len(self.dataspec.independent_state_columns)
        self.num_dependent = len(self.dataspec.dependent_state_columns)
        self.linear_independents = [layers.Dense(self.num_independent) for _ in range(2)]
        self.linear_dependent = layers.Dense(self.num_dependent)
