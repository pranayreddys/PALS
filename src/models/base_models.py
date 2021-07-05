import tensorflow as tf
from dataset.dataset import TimeSeriesDataset, TabularDataset


class BaseTimeSeriesModel(tf.keras.Model):

    def __init__(self,_dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__()
        self.dataspec = _dataspec
        

    def call(self, inputs):
        dependent_state_vals = _get_data(self.dataspec.dependent_state_columns, inputs)
        independent_state_vals = _get_data(self.dataspec.independent_state_columns, inputs)
        control_input_vals = _get_data(self.dataspec.control_input_columns, inputs)
        output = _predict_one_step(dependent_state_vals,independent_state_vals,control_input_vals)
        return output

    def _get_data(cols, inputs):
        #TODO: extracting relevant columns from inputs
        #simpler to return tensors but feel free to choose otherwise 
        #vals can be whatever we choose
        return vals

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
        # sets the model params
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
        # loss_values is a list of values corresponding to the loss functions in the config
        return loss_values

    def simple_predict(self,ts_data:TimeSeriesDataset,predict_config:PredictionConfig):
        # TODO: Same two models here as well. This prediction assumes that the ts_data 
        # actually has the control inputs (if any)  populated for the horizon 
        # need to raise an error if that is not true
        # output_data is a TimeSeriesDataset with the relevant columns having "_predicted" suffix
        return output_data


    def simple_impute(self,ts_data:TimeSeriesDataset,impute_config:ImputationConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here 
        return

    def simple_detect_outliers(self,ts_data:TimeSeriesDataset,outlier_config:OutlierDetectionConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        # We might prefer to use bidirectional models here as well
        return
    
    
class BaseControlSystemModel(BaseTimeSeriesModel):

    def __init__(self,_dataspec: TimeSeriesDataSpec):
        super(BaseTimeSeriesModel, self).__init__(_dataspec)
    
    def simple_control(self,ts_data:TimeSeriesDataset,control_config:TimeSeriesControlConfig):
        # TODO: LATER after other parts are done
        return control_vals


class BaseTransformationModel(tf.keras.Model):

    def __init__(self,_dataspec: TabularDataSpec):
        super(BaseTransformationModel, self).__init__()
        self.dataspec = _dataspec
        

    def call(self, inputs):
        dependent_vals = _get_data(self.dataspec.dependent_columns, inputs)
        independent_vals = _get_data(self.dataspec.independent_columns, inputs)
        output = _predict(dependent_vals,independent_vals)
        return output

    def _get_data(cols, inputs):
        #TODO: simpler to return tensors only here 
        return vals

    def _predict(independent_vals):
        # TODO: This is the main function to be implemented by the derived classes
        # inputs are tensors and output is a tensor
        # but internally we can convert the input tensors to pandas or whatever form is convenient
        # and do the computation with pandas and convert output back to tensor
        # if we are using keras models, then keeping it as tensors is simpler
        # alternate option insteaf of NotImplementedError is to use abstract methods

        raise NotImplementedError

    def simple_fit(self,ts_data:TabularDataset,train_config:TabularTrainingConfig):
        # TODO: This fit is primarily a wrapper around the keras model compile and fit
        # need to massage the dataset and config into appropriate form
        return


    def simple_evaluate(self,ts_data:TabularDataset,eval_config:TabularEvaluationConfig):
        # TODO: This evaluate is also a wrapper around the keras model functions
        return loss_values

    def simple_predict(self,ts_data:TabularDataset,predict_config:TabularPredictionConfig):
        # TODO: simple wrapper around predict
        # output_data is a TabularDataset with output column populated but with "_predicted" suffix
        return output_data


    def simple_impute(self,ts_data:TabularDataset,impute_config:TabularImputationConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        return

    def simple_detect_outliers(self,ts_data:TabularDataset,outlier_config:TabularOutlierDetectionConfig):
        # TODO: SKIP - not planning to implement unless we really need this
        return


class BaseGenerationModel:

    def __init__(self,_dataspec: TabularDataSpec):
        self.dataspec = _dataspec
        

    def simple_generate(self, generation_config:TabularDataGenerationConfig):
        # need to check if there is a match between the config and the dataspec
        # TODO: [Srujana] -- from the variable specs
        # will need additional methods
        return

