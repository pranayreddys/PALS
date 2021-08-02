from pydantic import BaseModel
from typing import Optional, Dict, List
from entities.key_entities import TimeSeriesDataSpec, TabularDataSpec
from entities.all_enums import OptimizerType, LossMetric, DistributionType, ModelClass, ColumnTransform
import tensorflow as tf
import tensorflow_probability as tfp


class TimeSeriesBaseConfig(BaseModel):
    """
    Class that describes training, evaluation and prediction configs.
    Args:
    model_class - Define the model class (example VAR/LSTM etc)
    model_parameters - Model specific parameters passed as dict
    context_window - Window size of points to be used as input for predicting/forecasting ahead.
    lead_gap - Gap between context window and forecasting window. Least value is 0.
    forecast_horizon - Window size for forecasting.
    """
    model_class: ModelClass
    model_parameters: Dict
    context_window: int 
    lead_gap: int 
    forecast_horizon: int 
    stride: int = 1
    model_save_folder: Optional[str]

class Optimizer(BaseModel):
    """
    The parameters to create the optimizer are defined here.
    The get_optimizer() function creates and returns the optimizer object
    according to the specifications in the config.
    Custom optimizers not implemented yet, added as a TODO item.
    """
    custom: Optional[bool] = False
    optimizer_type : OptimizerType
    # TODO: add support for custom training etc
    learning_rate: int = 0.001
    learning_rate_schedule : Optional[str] #TODO 

    def get_optimizer(self):
        if not self.custom:
            return eval("tf.keras.optimizers."+self.optimizer_type)(learning_rate = self.learning_rate)
        else:
            # To be filled for custom optimizers and learning rate schedules
            print("Custom optimizers not implemented")
            raise NotImplementedError



class Loss(BaseModel):
    """
    The parameters to create the loss are defined here.
    The get_loss() function creates and returns the loss object
    according to the specifications in the config. The definitions of the params:
    custom - Defines whether the loss is a default keras implemented loss or a custom loss.
    If custom is False, expected input is simply a loss type such as MeanSquaredError or MeanAbsoluteError.
    If custom is True, additional params such as column_weights, column_loss_type etc can be specified.
    get_loss() The get_loss object returns the appropriate loss function based on the config parameters.
    The class is primarily used in TimeSeriesTrainingConfig (defined below).
    """
    custom: Optional[bool] = False
    loss_type: Optional[LossMetric] = None
    column_weights: Optional[Dict[str, float]] = None
    column_loss_type: Optional[Dict[str, LossMetric]] = None
    is_column_aggregation: bool = True #TODO not implemented yet
    is_cell_aggregation: bool = True #TODO not implemented yet
    def get_loss(self, column_order: List[str] = None):
    #TODO: add more support for custom losses, also add more support for params to loss functions
        if not self.custom:
            assert self.loss_type, "loss_type required if not Custom Loss"
            return eval("tf.keras.losses."+self.loss_type)()
        else:
            assert column_order, "Column Order not provided"
            if self.loss_type:
                loss = eval("tf.keras.losses."+self.loss_type)()
            else:
                assert self.column_loss_type, "Provide columnwise dict[str, LossMetric if not providing single loss"
                self.column_loss_type = {k: eval("tf.keras.losses."+loss_type)() for k, loss_type in self.column_loss_type.items()}
            assert not((self.column_loss_type is not None) and (self.loss_type is not None)),"Both column loss type and loss type provided"
            column_order = {column: idx for idx, column in enumerate(column_order)}
            def ret_loss(y_true, y_pred):
                nonlocal loss
                l = 0
                if not self.column_weights:
                    self.column_weights = {k: 1 for k in column_order.keys()}
                for k,v in self.column_weights.items():
                    if not self.loss_type:
                        loss = self.column_loss_type[k]
                    l += (v*loss(y_true[:,:,column_order[k]], y_pred[:,:,column_order[k]]))
                return l
            return ret_loss

        


class TimeSeriesTrainingConfig(TimeSeriesBaseConfig):
    """
    The major config used for time series training - constructs the loss object, optimizer object, defines preprocessing
    configurations and also defines batchsize and epochs to be trained for.
    Args:
    search_space - TODO not used anywhere, a method to bound parameters for certain types of models
    search_paramters - TODO params required for constraining the search space.
    optimizer - Define the optimizer params mentioned in the Optimizer Class
    train_loss_function - Define Loss parameters mention in the Loss Class.
    metrics - List of simple loss metrics such as MeanSquaredError, MeanAbsoluteError
    batchsize - Batchsize for training.
    epochs - Number of epochs for training
    column_transformations - Transformations such as StandardScaling, MinMaxScaling, OneHotEncoding.
    """
    search_space: Optional[Dict] #TODO
    search_parameters: Optional[Dict] #TODO
    optimizer: Optimizer
    train_loss_function: Loss
    epochs: int
    metrics: List[LossMetric] = None
    batchsize: int
    column_transformations: Dict[str, ColumnTransform] = None

    def get_optimizer(self):
        return self.optimizer.get_optimizer()
    
    def get_loss(self, column_order=None):
        return self.train_loss_function.get_loss(column_order)
    
    def get_metrics(self):
        return [eval("tf.keras.losses."+loss)() for loss in self.metrics]

class TimeSeriesEvaluationConfig(TimeSeriesBaseConfig):
    """
    Not implemented yet, would involve counterfactual evaluation etc
    """
    #TODO: use this config in simple_evaluate
    loss_list: List[LossMetric]
    def get_losses(self):
        return [("tf.keras.losses."+loss)() for loss in self.loss_list]
    

class TimeSeriesPredictionConfig(TimeSeriesBaseConfig):
    """
    Additional Params not implemented, supports all the base parameters of TimeSeriesBaseConfig.
    """
    pass

class TimeSeriesImputationConfig(TimeSeriesBaseConfig):
    """
    TODO: Imputation not implemented
    """
    pass

class TimeSeriesOutlierDetectionConfig(TimeSeriesBaseConfig):
    """
    TODO: Outlier detection not implemented
    """
    pass

class TimeSeriesControlConfig(TimeSeriesBaseConfig):
    """
    TODO
    """
    pass

class TabularBaseConfig(BaseModel):
    """
    Config for tabular data, such as user embeddings.
    """
    data_spec: TabularDataSpec
    model_class: ModelClass 
    model_parameters: Dict
    column_transformations: Dict[str, ColumnTransform] = None

class TabularTrainingConfig(TabularBaseConfig):
    """
    Training config for tabular data. TODO: Needs to be tested.
    """
    search_space: Dict
    search_parameters: Dict
    optimizer: Optimizer
    train_loss_function: Loss
    epochs: int
    callbacks: List[str]
    batchsize: int
    def get_optimizer(self):
        self.optimizer.get_optimizer()
    
    def get_loss(self):
        self.loss.get_loss()
    

"""
The rest of the classes below are not implemented yet, support can be added when required.
"""
class TabularEvaluationConfig(TabularBaseConfig):
    pass

class TabularPredictionConfig(TabularBaseConfig):
    pass

class TabularImputationConfig(TabularBaseConfig):
    pass

class TabularPredictionConfig(TabularBaseConfig):
    pass

class TabularOutlierDetectionConfig(TabularBaseConfig):
    pass

class TabularDataGenerationConfig(TabularBaseConfig):
    pass