import tensorflow as tf
from dataset.dataset import TimeSeriesDataset, TabularDataset


class BaseTimeSeriesModel(tf.keras.Model):

    def __init__(self,_dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__()
        self.dataspec = _dataspec
        

    def call(self, inputs):
        dependent_state_vals = _get_data(self.dataspec.dependent_state_columns, inputs)
        independent_state_vals = _get_data(self.dataspec.dependent_state_columns, inputs)
        control_input_vals = _get_data(self.dataspec.dependent_state_columns, inputs)
        output = _predict_one_step(dependent_state_vals,independent_state_vals,control_input_vals)
        return output

    def _get_data(cols, inputs):
        #TODO: simpler to return tensors but feel free to choose otherwise 
        return

    def _predict_one_step(dependent_state_vals,independent_state_vals,control_input_vals):
        # This is the main function to be implemented by the derived classes
        # inputs are tensors and output is a tensor
        # but internally we can convert the input tensors to pandas or whatever form is convenient
        # and do the computation with pandas and convert output back to tensor
        # if we are using keras models, then keeping it as tensors is simpler
        # alternate option insteaf of NotImplementedError is to use abstract methods

        raise NotImplementedError

    def simple_fit(self,ts_data:TimeSeriesDataset,train_config:TrainingConfig):
        # TODO: This fit is primarily a wrapper around the keras model compile and fit
        # need to massage the dataset and config into appropriate form
        # There are two modes of computing accuracy: fixed lead or fixed anchor index + horizon
        # which can be used to fit the model - we can implement the latter one first
        return


    def simple_evaluate(self,ts_data:TimeSeriesDataset,eval_config:EvaluationConfig):
        # TODO: This evaluate is also a wrapper around the keras model functions
        # Again two models: fixed lead or anchor time + horizon
        # In the fixed anchor time case, we use that point's (or before) observations
        # to generate predictions for the future horizon and then evaluate the accuracy of 
        # the ground truth vs. the predicted values. 
        # In the fixed lead (say l), for every time point t that needs to be evaluated
        # we use the model to make predictions at t based on known observations at time (t-l) 
        # (or before) and then evaluate the accuracy
        return

    def simple_predict(self,ts_data:TimeSeriesDataset,predict_config:PredictionConfig):
        # TODO: Same two models here as well. This prediction assumes that the ts_data 
        # actually has the control inputs (if any)  populated for the horizon 
        # need to raise an error if that is not true
        return


    def simple_impute(self,ts_data:TimeSeriesDataset,impute_config:ImputationConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here 
        return

    def simple_detect_outliers(self,ts_data:TimeSeriesDataset,outlier_config:OutlierDetectionConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here as well
        return
