import tensorflow as tf
import numpy as np
from dataset.dataset import TimeSeriesDataset, TabularDataset
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec
from entities.modeling_configs import *
import models.simple_transformations as simple_transform
import os
import random
import abc
import pandas as pd
from typing import Tuple
"""
In this file, the BaseTimeSeriesModel is the only model that has been tested. Also implemented TabularModel,
but it has never been tested so might be prone to bugs.
"""

class BaseTimeSeriesModel(tf.keras.Model, abc.ABC):
    """
    This abstract class is the parent class for all time series models. The base class
    unrolls time series step by step using the _predict_one_step function. For example,
    given X[1:T], the time series model will compute X[T+1] = _predict_one_step(X[1:T])
    This input is then fed back to get X[T+2] = _predict_one_step(X[2:(T+1)]) and so on.
    Assumptions:
    1. The inputs are all in numeric or categorical format. In case the input is provided as
    categorical format, the input transformation needs to be specified in column transformations.
    2. Basic model assumed C[0] -> S[0]
                            |       |
                           C[1] -> S[1]
                           |        | 
                           C[2] -> S[2]
        and so on, where C denotes control and S denotes state.
        Special care needs to be taken in indexing control, since C[0:1], S[0] will be used to predict S[1]
    3. There is a single unique ID column for each time series in the dataset
    4. In the code written below, control and series attribute columns are treated quite similarly. In case
    internally we want to perform something different depending on series attribute columns, it needs to be
    done within the _predict function of the child class that is being implemented (refer models/models.py)
    """

    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__()
        self.dataspec = _dataspec.copy(deep=True)
        self.preprocessor = None # Will be initialized in the fit

    def call(self, inputs):
        """
        Implements the call method used by keras models. Internally, the _predict_one_step
        method is called to compute prediction one day into the future 
        """

        controls, states = inputs
        
        # controls = tf.convert_to_tensor(controls) 
        
        # states = tf.convert_to_tensor(states)
        # Note that state only has the shape batch x context_window x state_feature_size

        predictions = []
        for horizon_step in range(self.config.lead_gap + self.config.forecast_horizon):
            new_state = self._predict_one_step(states, 
                                controls[:, horizon_step: self.config.context_window+1+horizon_step])
            
            
            if horizon_step >= self.config.lead_gap: 
                predictions.append(new_state)

            states = tf.concat((states[:, 1:], tf.reshape(new_state,[new_state.shape[0], 1,-1])), axis=1)
            # Input 1 to the function has shape B x T-1 x S
            # Input 2 has shape B x S, so needs to be reshaped to B x 1 x S and stacked
            # In pseudocode form, the above line is equivalent to X[1:T] = X[2:T+1]

        predictions = tf.stack(predictions)
        # T x B x S

        predictions = tf.transpose(predictions, [1, 0, 2])
        # B x T x S

        return predictions

    def _split_window(self, timeseries, config):
        """
        Function that breaks a time series window that consists of both control and state inputs
        into the respective components. The control and state horizons are of different length
        since control is under model control unlike state.
        """
        dim_controls = len(self.dataspec.control_input_columns) + len(self.dataspec.series_attribute_columns) \
                         if (len(self.dataspec.control_input_columns) + len(self.dataspec.series_attribute_columns))>0 else 1
        controls = timeseries[:, :, 0:dim_controls]
        states = timeseries[:,:,dim_controls:]
        return (tf.convert_to_tensor(controls), tf.convert_to_tensor(states[:,0:config.context_window])), \
                tf.convert_to_tensor(states[:,(config.context_window+config.lead_gap):config.context_window+config.lead_gap+config.forecast_horizon])

    def _make_subset(self,timeseries, config):
        """
        Dataloader preparation in the control, state format for a single time series
        """
        assert(config.forecast_horizon>=1)
        assert(config.context_window>=1)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=timeseries,
            targets=None,
            sequence_length=config.context_window
                            + config.lead_gap
                            + config.forecast_horizon,
            sequence_stride=config.stride,
            shuffle=False,
            batch_size=config.batchsize) # returns batch x time x feature

        ds = ds.map(lambda x: self._split_window(x, config)) 
        return ds

    def _make_dataset(self, ts_data: TimeSeriesDataset, config):
        """
        For the problem that we are interested in, there are multiple time series that are determined
        by the ID of the 
        """
        dataset_subsets = []
        i = 0
        dataset_order = []
        for _, grouped_subset in ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns 
                                        + self.dataspec.series_attribute_columns)
            state_subset = self._get_data(grouped_subset, self.dataspec.independent_state_columns
                                                + self.dataspec.dependent_state_columns)

            dataset_subsets.append(self._make_subset(np.concatenate((control_subset, state_subset), axis=1), config))
            for _ in range(len(dataset_subsets[i])):
                dataset_order.append(i)
            i += 1
        
        random.shuffle(dataset_order)
        dataset_order = np.array(dataset_order, dtype=np.int64)
        return tf.data.experimental.choose_from_datasets(dataset_subsets,
                tf.data.Dataset.from_tensor_slices(dataset_order))

    @staticmethod
    def _get_data(inputs, cols):
        """
        If cols is left blank, returns a dummy array (useful when some columns are not present, for
        example control columns)
        """
        if not cols:
            return np.array(list(range(len(inputs)))).reshape(-1,1)
        return inputs[cols].values.reshape(len(inputs),-1)

    @abc.abstractmethod
    def _predict_one_step(self, state_vals, control_input_vals):
        """This is the main function to be implemented by the derived classes
        inputs are tensors and output is a tensor
        but internally we can convert the input tensors to pandas or whatever form is convenient
        and do the computation with pandas and convert output back to tensor
        if we are using keras models, then keeping it as tensors is simpler. 
        """
        pass


    def _build_model(self, train_config: TimeSeriesTrainingConfig):
        """
        Compiles and builds a keras eager model, with the correct loss, optimizer, and
        metrics
        """
        self.compile(loss=train_config.get_loss(self.dataspec.independent_state_columns
                                    + self.dataspec.dependent_state_columns),
                    optimizer=train_config.get_optimizer(), 
                    metrics=train_config.get_metrics(),
                    run_eagerly=True)

    def simple_fit(self, train_data: TimeSeriesDataset, 
                        val_data: TimeSeriesDataset,
                        train_config: TimeSeriesTrainingConfig,
                        model_params):
        """This fit is primarily a wrapper around the keras model compile and fit
        need to massage the dataset and config into appropriate form
        """
        self.config = train_config
        self.preprocessor = simple_transform.SimplePreprocessModel(column_transformations=self.config.column_transformations)
        train_data= self.preprocessor.simple_fit_predict(train_data)
        self._build_model(train_config)
        self.dataspec = train_data.dataset_spec
        self.dataspec.control_input_columns.remove('nudge_0') #HACK remove this line
        self.set_params(model_params)
        dataset = self._make_dataset(train_data, train_config)
        history = self.fit(dataset,epochs=train_config.epochs, shuffle=True)
        # Add verbose=0 here to stop output
        return history # Log data returned

    def save_model(self, model_save_folder):
        """
        Used for saving model and end of training.
        Currently no method for checkpointing models epoch by epoch, but can be extended
        by using custom callbacks along with this method. Also saves the simple preprocessor
        model
        """
        if not os.path.isdir(model_save_folder):
            os.makedirs(model_save_folder)
        
        if not os.path.isdir(os.path.join(model_save_folder, "sklearn")):
            os.makedirs(os.path.join(model_save_folder, "sklearn"))
        
        if not os.path.isdir(os.path.join(model_save_folder, "keras")):
            os.makedirs(os.path.join(model_save_folder, "keras"))
        
        self.preprocessor.save(os.path.join(model_save_folder, "sklearn/preprocessor.pickle"))
        self.save_weights(os.path.join(model_save_folder,"keras", "weights"), save_format='tf')

    def load_model(self, model_save_folder, train_config, dataset):
        """
        Loads a saved model given a file path and the original training config.
        """
        self.preprocessor = simple_transform.SimplePreprocessModel()
        self.dataspec = self.preprocessor.load(os.path.join(model_save_folder, "sklearn/preprocessor.pickle"))
        self._build_model(train_config)
        self.config = train_config
        
        #Reason for below line https://stackoverflow.com/questions/63658086/tensorflow-2-0-valueerror-while-loading-weights-from-h5-file
        self.simple_predict(dataset, train_config)
        self.load_weights(os.path.join(model_save_folder, "keras", "weights"))

    def simple_evaluate(self, ts_data: TimeSeriesDataset, eval_config: TimeSeriesEvaluationConfig):
        """
        Simple wrapper around the keras evaluate function. Outputs the metrics that were specified during
        training. For additonal/more complex metrics, further evaluation functions need to be defined, along
        with calls to simple_predict.
        """
        ts_data = self.preprocessor.simple_predict(ts_data)
        self.config= eval_config

        per_series= {}
        for id, grouped_subset in ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns + self.dataspec.series_attribute_columns)
            state_columns = self.dataspec.independent_state_columns + self.dataspec.dependent_state_columns
            state_subset = self._get_data(grouped_subset, state_columns)
            per_series[id]= self.evaluate(self._make_subset(np.concatenate((control_subset, state_subset), axis=1), eval_config), return_dict=True)

        eval_results = {}
        for _id, eval_dict in per_series.items():
            for k,value in eval_dict.items():
                eval_results[("eval_"+str(_id)+"_"+str(k))] = value 
        return eval_results

    def simple_predict(self, ts_data_orig: TimeSeriesDataset, predict_config: TimeSeriesPredictionConfig):
        """
        Returns a forecast_horizon x state vector for each prediction point. Data loading is performed
        similar to train.
        """
        self.config = predict_config
        ts_data = self.preprocessor.simple_predict(ts_data_orig)
        ret_ts_data = TimeSeriesDataset(ts_data.dataset_spec, blank_dataset=True)
        ret_ts_data.data = ts_data.data.copy()
        state_columns = self.dataspec.independent_state_columns + self.dataspec.dependent_state_columns
        predicted_columns = []
        inversion_mapping = {}
        total_window_size= self.config.context_window + self.config.lead_gap + self.config.forecast_horizon
        for horizon_step in range(self.config.forecast_horizon):
            for state in state_columns:
                predict_column_name = state+"_horizon_"+str(horizon_step+1)+"_predict"
                predicted_columns.append(predict_column_name)
                ret_ts_data.data[predict_column_name] = np.NaN
                inversion_mapping[predict_column_name] = state 
        for key, grouped_subset in ret_ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns + self.dataspec.series_attribute_columns)
            state_subset = self._get_data(grouped_subset, state_columns)
            predictions = self.predict(self._make_subset(np.concatenate((control_subset, state_subset), axis=1), predict_config)) # B x T x S
            predictions = predictions.reshape(predictions.shape[0], -1) # B x T x S -> B x (T*S)
            assert(predictions.shape[0]==(grouped_subset.shape[0]-total_window_size+1))
            self.preprocessor.invert(predictions, predicted_columns, inversion_mapping)
            predictions_corrected_shape = np.full((grouped_subset.shape[0],predictions.shape[1]),np.NaN)
            predictions_corrected_shape[:predictions.shape[0], :] = predictions
            ts_data_orig.assign_id_vals(key, predicted_columns, predictions_corrected_shape)
        return ts_data_orig


    def simple_impute(self, ts_data: TimeSeriesDataset, impute_config: TimeSeriesImputationConfig):
        """TODO: SKIP - not planning to implement unless we really need this
        We might prefer to use bidirectional models here """
        return

    def simple_detect_outliers(self, ts_data: TimeSeriesDataset, outlier_config: TimeSeriesOutlierDetectionConfig):
        """TODO: SKIP - not planning to implement unless we really need this
        We might prefer to use bidirectional models here as well"""
        return
    
    @abc.abstractmethod
    def visualize(self, folder_path: str):
        pass
    
    @abc.abstractmethod
    def set_params(self, params):
        """
        Abstract method used to specify model specific parameters
        """
        pass
    
class BaseControlSystemModel(BaseTimeSeriesModel):

    def __init__(self,_dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__(_dataspec)
    
    def simple_control(self, ts_data: TimeSeriesDataset, control_config: TimeSeriesControlConfig):
        # TODO: LATER after other parts are done
        pass
        #return control_vals


class BaseTransformationModel(tf.keras.Model, abc.ABC):

    def __init__(self, _dataspec: TabularDataSpec):
        super(BaseTransformationModel, self).__init__()
        self.dataspec = _dataspec
        

    def call(self, inputs):
        output = _predict(inputs[:, :len(self.dataspec.independent_columns)],
                         inputs[:, len(self.dataspec.independent_columns):])
        return output

    @staticmethod
    def _get_data(inputs: pd.DataFrame, cols: List[str]):
        #Returning numpy arrays
        return inputs[cols].values

    @abc.abstractmethod
    def _predict(self, independent_vals, dependent_vals):
        """This is the main function to be implemented by the derived classes
        inputs are tensors and output is a tensor
        but internally we can convert the input tensors to pandas or whatever form is convenient
        and do the computation with pandas and convert output back to tensor
        if we are using keras models, then keeping it as tensors is simpler
        alternate option insteaf of NotImplementedError is to use abstract methods"""
        pass

    def simple_fit(self, tb_data: TabularDataset, train_config: TabularTrainingConfig):
        self.compile(loss=train_config.get_loss(),
                        optimizer=train_config.get_optimizer(),
                        metrics=train_config.get_metrics())
        columns = self.dataspec.independent_columns + self.dataspec.dependent_columns 
        history = self.fit(x=self._get_data(tb_data.data, columns), 
                            y =self._get_data(tb_data.data, self.dataspec.target_columns), 
                            epochs=train_config.epochs, 
                            shuffle=True
                        )
        return


    def simple_evaluate(self, tb_data: TabularDataset, eval_config: TabularEvaluationConfig):
        # TODO: Parse tabularevaluationconfig and pass arguments to evaluate.
        columns = self.dataspec.independent_columns + self.dataspec.dependent_columns 
        return self.evaluate(x=self._get_data(columns, tb_data.data),
                            y=self._get_data(self.dataspec.target_columns, tb_data.data))

    def simple_predict(self, tb_data: TabularDataset, predict_config: TabularPredictionConfig):
        columns = self.dataspec.independent_columns + self.dataspec.dependent_columns
        output_data = TabularDataset(tb_data.dataset_spec, blank_dataset=True)
        output = self.predict(self._get_data(columns, tb_data.data))
        for index, target in enumerate(self.dataspec.target_columns):
            output_data[target+"_predict"] = output[:, index]
        return output_data


    def simple_impute(self, ts_data: TabularDataset, impute_config: TabularImputationConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        return

    def simple_detect_outliers(self, ts_data: TabularDataset, outlier_config: TabularOutlierDetectionConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        return


class BaseGenerationModel:

    def __init__(self, _dataspec: TabularDataSpec):
        self.dataspec = _dataspec
        

    def simple_generate(self, generation_config: TabularDataGenerationConfig):
        # need to check if there is a match between the config and the dataspec
        # TODO: [Srujana] -- from the variable specs
        # will need additional methods
        return

class BaseTabularPredictor(abc.ABC):
    """The base abstract class used by nudge optimizers, reward predictors and estimators.
    This class may be making some of the other classes such as BaseTransformationModel a bit redundant, may need to be fixed
    in the future. 
    """
    def __init__(self, _dataspec):
        """Initialize dataspec

        Args:
            _dataspec (NudgeOptimizationDataSpec): Since it is used primarily for the above models
             
        """
        self._dataspec = _dataspec
    
    @abc.abstractmethod
    def _predict(self, state: np.array):
        """ Needs to return predicted columns from numpy array"""
        pass

    @abc.abstractmethod
    def set_params(self, params: Dict):
        """
        Meant for initialization parameters (for example model loading in reward predictors)
        """
        pass
    
    @staticmethod
    def _get_data(inputs: pd.DataFrame, cols: List[str]):
        return inputs[cols].values
    