import tensorflow as tf
from dataset.dataset import TimeSeriesDataset, TabularDataset
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec
from entities.modeling_configs import TimeSeriesTrainingConfig


class BaseTimeSeriesModel(tf.keras.Model):
#ASSUMPTION: All inputs are given in numeric format
#ASSUMPTION 2: Output is assumed to be the next state of the time series
#TODO: Add second dataloader that handles each time series as different points.  

    def __init__(self, _dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__()
        self.dataspec = _dataspec

    def call(self, inputs):
        # state_vals = _get_data(self.dataspec.dependent_state_columns 
        #                         + self.dataspec.independent_state_columns, inputs)
        # control_input_vals = _get_data(self.dataspec.control_input_columns, inputs)
        # output = _predict_one_step(state_vals,control_input_vals)
        
        controls, states = inputs
        predictions = []
        for horizon_step in range(self.dataspec.forecast_horizon):
            new_state = self._predict_one_step(states, controls)
            predictions.append(new_state)
            states[:, 0 : states.shape[1]-1, :]=  states[:, 1:, :]
            # batch x time x features 
            
            states[:, states.shape[1]-1, :] = new_state

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    @staticmethod
    def _split_window(controls, states, config):
        input_slice= slice(0, config.context_window)
        label_slice= slice(config.context_window+config.lead_gap, None)
        return (controls, states[:,input_slice,:]), states[:,label_slice,:]

    @staticmethod
    def _make_subset(controls, states, config):
        #TODO: change the params available here
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=controls,
            targets=states,
            sequence_length=config.context_window
                            + config.lead_gap
                            + config.forecast_horizon,
            sequence_stride=config.stride,
            shuffle=False,
            batch_size=config.batchsize,) # returns batch x time x feature

        ds = ds.map(lambda x, y: BaseTimeSeriesModel._split_window(x,y, config)) # FIXME: Slightly hacky
        return ds

    def _make_dataset(self, ts_data: TimeSeriesDataset, config):
        dataset_subsets = []
        for _, grouped_subset in ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns)
            state_subset = self._get_data(grouped_subset, self.dataspec.independent_state_columns
                                                + self.dataspec.dependent_state_columns)

            dataset_subsets.append(_make_subset(control_subset, state_subset, config))
        return tf.data.experimental.choose_from_datasets(dataset_subsets,
                                        tf.data.Dataset.range(len(dataset_subsets)))
    @staticmethod
    def _get_data(cols, inputs):
        #Extracting relevant columns from inputs
        #Returns numpy array 
        return inputs[cols].values

    def _predict_one_step(self, state_vals, control_input_vals):
        # This is the main function to be implemented by the derived classes
        # inputs are tensors and output is a tensor
        # but internally we can convert the input tensors to pandas or whatever form is convenient
        # and do the computation with pandas and convert output back to tensor
        # if we are using keras models, then keeping it as tensors is simpler
        # alternate option insteaf of NotImplementedError is to use abstract methods

        raise NotImplementedError

    def simple_fit(self, train_data: TimeSeriesDataset, 
                        val_data: TimeSeriesDataSet,
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
        self.compile(loss=tf.losses.MeanSquaredError(), #TODO: Fix loss function metric
                        optimizer=train_config.get_optimizer(),
                        metrics=[])
        history = self.fit(_make_dataset(train_data, train_config), 
                            epochs=train_config.epochs, shuffle=False
                        )
                            # validation_data=window.val,
        return


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
        for _, grouped_subset in ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns)
            state_columns = self.dataspec.independent_state_columns + self.dataspec.dependent_state_columns
            state_subset = self._get_data(grouped_subset, state_columns)
            #TODO: Add params for evaluate
            self.evaluate(_make_subset(control_subset, state_subset, eval_config))

    def simple_predict(self, ts_data: TimeSeriesDataset, predict_config: TimeSeriesPredictionConfig):
        # TODO: Same two models here as well. This prediction assumes that the ts_data 
        # actually has the control inputs (if any)  populated for the horizon 
        # need to raise an error if that is not true
        # output_data is a TimeSeriesDataset with the relevant columns having "_predicted" suffix
        ret_ts_data = TimeSeriesDataset(ts_data.dataset_spec, blank_dataset=True)
        ret_ts_data.data = ts_data.data.copy()
        for _, grouped_subset in ret_ts_data.subset_per_id():
            control_subset = self._get_data(grouped_subset, self.dataspec.control_input_columns)
            state_columns = self.dataspec.independent_state_columns + self.dataspec.dependent_state_columns
            state_subset = self._get_data(grouped_subset, state_columns)
            for state in state_columns:
                grouped_subset[state+"_predict"] = -1
            #TODO: What is the output of predict? We are predicting many values, many times for each day
            self.predict(_make_subset(control_subset, state_subset, predict_config))
        return ret_ts_data


    def simple_impute(self, ts_data: TimeSeriesDataset, impute_config: TimeSeriesImputationConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here 
        return

    def simple_detect_outliers(self, ts_data: TimeSeriesDataset, outlier_config: TimeSeriesOutlierDetectionConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here as well
        return
    
    
class BaseControlSystemModel(BaseTimeSeriesModel):

    def __init__(self,_dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__(_dataspec)
    
    def simple_control(self, ts_data: TimeSeriesDataset, control_config: TimeSeriesControlConfig):
        # TODO: LATER after other parts are done
        return control_vals


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
    def _get_data(cols, inputs):
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
        # TODO: This fit is primarily a wrapper around the keras model compile and fit
        # need to massage the dataset and config into appropriate form
        self.compile(loss=tf.losses.MeanSquaredError(), #TODO: Fix loss function metric
                        optimizer=train_config.get_optimizer(),
                        metrics=[])
        columns = self.dataspec.dependent_columns + self.dataspec.independent_columns
        history = self.fit(x=self._get_data(columns, tb_data.data), 
                            y =self._get_data(self.dataspec.target_columns, tb_data.data), 
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
        # TODO: simple wrapper around predict
        # output_data is a TabularDataset with output column populated but with "_predicted" suffix
        columns = self.dataspec.dependent_columns + self.dataspec.independent_columns
        output_data = TabularDataset(tb_data.dataset_spec, blank_dataset=True)
        for index, target in enumerate(self.dataspec.target_columns):
            output_data[target+"_predict"] = output_data[:, index]
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

