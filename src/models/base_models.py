import tensorflow as tf
import numpy as np
from dataset.dataset import TimeSeriesDataset, TabularDataset
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec
from entities.modeling_configs import *


class BaseTimeSeriesModel(tf.keras.Model):
#ASSUMPTION: All inputs are given in numeric format
#ASSUMPTION 2: Output is assumed to be the next state of the time series
#TODO: Add second dataloader that handles each time series as different points.  

    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__()
        self.dataspec = _dataspec.copy(deep=True)

    def call(self, inputs):
        controls, states = inputs
        controls = tf.convert_to_tensor(controls)
        states = tf.convert_to_tensor(states)
        predictions = []
        for horizon_step in range(self.config.lead_gap + self.config.forecast_horizon):
            new_state = self._predict_one_step(states, 
                                controls[horizon_step: self.config.context_window+1+horizon_step])
            if horizon_step >= self.config.lead_gap:
                predictions.append(new_state)
            # states[:, 0 : states.shape[1]-1]=  states[:, 1:]
            # states[:, states.shape[1]-1] = new_state
            states = tf.concat((states[:, 1:], tf.reshape(new_state,[new_state.shape[0], 1,-1])), axis=1)
            # Input 1 to the function has shape B x T-1 x S
            # Input 2 has shape B x S, so needs to be reshaped to B x 1 x S and stacked

        predictions = tf.stack(predictions)
        # T x B x S

        predictions = tf.transpose(predictions, [1, 0, 2])
        # B x T x S

        return predictions

    def _split_window(self, timeseries, config):
        dim_controls = len(self.dataspec.control_input_columns) if self.dataspec.control_input_columns else 1
        controls = timeseries[:, :, 0:dim_controls]
        states = timeseries[:,:,dim_controls:]
        return (controls, states[:,0:config.context_window]), \
                states[:,(config.context_window+config.lead_gap):config.context_window+config.lead_gap+config.forecast_horizon]

    def _make_subset(self,timeseries, config):
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
        dataset_subsets = []
        for _, grouped_subset in ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns)
            state_subset = self._get_data(grouped_subset, self.dataspec.independent_state_columns
                                                + self.dataspec.dependent_state_columns)

            dataset_subsets.append(self._make_subset(np.concatenate((control_subset, state_subset), axis=1), config))
        
        return dataset_subsets[0] #FIXME
        # return tf.data.experimental.choose_from_datasets(dataset_subsets,
        #                                 tf.data.Dataset.range(len(dataset_subsets)))
    @staticmethod
    def _get_data(inputs, cols):
        if not cols:
            return np.array(list(range(len(inputs)))).reshape(-1,1)
        return inputs[cols].values.reshape(len(inputs),-1)

    def _predict_one_step(self, state_vals, control_input_vals):
        # This is the main function to be implemented by the derived classes
        # inputs are tensors and output is a tensor
        # but internally we can convert the input tensors to pandas or whatever form is convenient
        # and do the computation with pandas and convert output back to tensor
        # if we are using keras models, then keeping it as tensors is simpler
        # alternate option insteaf of NotImplementedError is to use abstract methods

        raise NotImplementedError

    def simple_fit(self, train_data: TimeSeriesDataset, 
                        val_data: TimeSeriesDataset,
                        train_config: TimeSeriesTrainingConfig):
        # TODO: This fit is primarily a wrapper around the keras model compile and fit
        # need to massage the dataset and config into appropriate form
        # There are two modes of computing accuracy: fixed lead or fixed anchor index + horizon
        # which can be used to fit the model - we can implement the latter one first
        # sets the model params
        # ASSUMPTION: TimeSeriesDataset is the train split.
        # control_data = self._get_data(train_data.data, self.dataspec.control_input_columns)
        # state_data = self._get_data(train_data.data, self.dataspec.independent_state_columns
        #                                         + self.dataspec.dependent_state_columns)
        
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                             patience=patience,
        #                                             mode='min')
        self.config = train_config
        self.compile(loss=train_config.get_loss(), optimizer=train_config.get_optimizer(), 
                    metrics=train_config.get_metrics(),run_eagerly=True)
        dataset = self._make_dataset(train_data, train_config)
        history = self.fit(dataset,  verbose=1,
                            epochs=train_config.epochs, shuffle=False
                        )
        return history


    def simple_evaluate(self, ts_data: TimeSeriesDataset, eval_config: TimeSeriesEvaluationConfig):
        # TODO: This evaluate is also a wrapper around the keras model functions
        # Again two models: fixed lead or anchor time + horizon
        # In the fixed anchor time case, we use that point's (or before) observations
        # to generate predictions for the future horizon and then evaluate the accuracy of 
        # the ground truth vs. the predicted values. 
        # In the fixed lead (say l), for every time point t that needs to be evaluated
        # we use the model to make predictions at t based on known observations at time (t-l) 
        # (or before) and then evaluate the accuracy
        # loss_values is a list of values corresponding to the loss functions in the config
        self.config= eval_config
        per_series= []
        for _, grouped_subset in ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns)
            state_columns = self.dataspec.independent_state_columns + self.dataspec.dependent_state_columns
            state_subset = self._get_data(grouped_subset, state_columns)
            per_series.append(self.evaluate(self._make_subset(np.concatenate((control_subset, state_subset), axis=1), eval_config)))
        
        return per_series

    def simple_predict(self, ts_data: TimeSeriesDataset, predict_config: TimeSeriesPredictionConfig):
        # TODO: Same two models here as well. This prediction assumes that the ts_data 
        # actually has the control inputs (if any)  populated for the horizon 
        # need to raise an error if that is not true
        # output_data is a TimeSeriesDataset with the relevant columns having "_predicted" suffix
        self.config = predict_config
        ret_ts_data = TimeSeriesDataset(ts_data.dataset_spec, blank_dataset=True)
        ret_ts_data.data = ts_data.data.copy()
        state_columns = self.dataspec.independent_state_columns + self.dataspec.dependent_state_columns
        predicted_columns = []
        total_window_size= self.config.context_window + self.config.lead_gap + self.config.forecast_horizon
        for horizon_step in range(self.config.forecast_horizon):
            for state in state_columns:
                predict_column_name = state+"_horizon_"+str(horizon_step+1)+"_predict"
                predicted_columns.append(predict_column_name)
                ret_ts_data.data[predict_column_name] = np.NaN
        for key, grouped_subset in ret_ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns)
            state_subset = self._get_data(grouped_subset, state_columns)
            predictions = self.predict(self._make_subset(np.concatenate((control_subset, state_subset), axis=1), predict_config)) # B x T x S
            predictions = predictions.reshape(predictions.shape[0], -1) # B x T x S -> B x (T*S)
            assert(predictions.shape[0]==(grouped_subset.shape[0]-total_window_size+1))
            predictions_corrected_shape = np.full((grouped_subset.shape[0],predictions.shape[1]),np.NaN)
            predictions_corrected_shape[:predictions.shape[0], :] = predictions
            ret_ts_data.assign_id_vals(key, predicted_columns, predictions_corrected_shape)
        return ret_ts_data


    def simple_impute(self, ts_data: TimeSeriesDataset, impute_config: TimeSeriesImputationConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here 
        return

    def simple_detect_outliers(self, ts_data: TimeSeriesDataset, outlier_config: TimeSeriesOutlierDetectionConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here as well
        return
    
    def set_params(self, params):
        ## Needs to be overloaded
        pass
    
class BaseControlSystemModel(BaseTimeSeriesModel):

    def __init__(self,_dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__(_dataspec)
    
    def simple_control(self, ts_data: TimeSeriesDataset, control_config: TimeSeriesControlConfig):
        # TODO: LATER after other parts are done
        pass
        #return control_vals


class BaseTransformationModel(tf.keras.Model):

    def __init__(self, _dataspec: TabularDataSpec):
        super(BaseTransformationModel, self).__init__()
        self.dataspec = _dataspec
        

    def call(self, inputs):
        # dependent_vals = _get_data(self.dataspec.dependent_columns, inputs)
        # independent_vals = _get_data(self.dataspec.independent_columns, inputs)
        output = _predict(inputs[:, :len(self.dataspec.dependent_columns)],
                         inputs[:, len(self.dataspec.dependent_columns):])
        return output

    @staticmethod
    def _get_data(inputs, cols):
        #Returning numpy arrays
        return inputs[cols].values

    def _predict(self, independent_vals):
        # TODO: This is the main function to be implemented by the derived classes
        # inputs are tensors and output is a tensor
        # but internally we can convert the input tensors to pandas or whatever form is convenient
        # and do the computation with pandas and convert output back to tensor
        # if we are using keras models, then keeping it as tensors is simpler
        # alternate option insteaf of NotImplementedError is to use abstract methods

        raise NotImplementedError

    def simple_fit(self, tb_data: TabularDataset, train_config: TabularTrainingConfig):
        self.compile(loss=train_config.get_loss(),
                        optimizer=train_config.get_optimizer(),
                        metrics=[])
        columns = self.dataspec.dependent_columns + self.dataspec.independent_columns
        history = self.fit(x=self._get_data(tb_data.data, columns), 
                            y =self._get_data(tb_data.data, self.dataspec.target_columns), 
                            epochs=train_config.epochs, 
                            shuffle=True
                        )
        return


    def simple_evaluate(self, tb_data: TabularDataset, eval_config: TabularEvaluationConfig):
        # TODO: Parse tabularevaluationconfig and pass arguments to evaluate.
        columns = self.dataspec.dependent_columns + self.dataspec.independent_columns
        return self.evaluate(x=self._get_data(columns, tb_data.data),
                            y=self._get_data(self.dataspec.target_columns, tb_data.data))

    def simple_predict(self, tb_data: TabularDataset, predict_config: TabularPredictionConfig):
        columns = self.dataspec.dependent_columns + self.dataspec.independent_columns
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

