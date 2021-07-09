from pydantic import BaseModel
from typing import Optional
from entities.key_entities import TimeSeriesDataSpec, TabularDataSpec, LossFunctionSpec
from entities.all_enums import OptimizerType, LossMetric
import tensorflow as tf

class TimeSeriesBaseConfig(BaseModel):
    data_spec: TimeSeriesDataSpec
    model_class: ModelClass # TODO: need to create this
    model_parameters: dict
    context_window: int 
    lead_gap: int 
    forecast_horizon: int 
    stride: int
    output_file_prefix: str = None
    output_dir: str = None

class Optimizer(BaseModel):
    custom: Optional[bool] = False
    optimizer_type : OptimizerType
    # TODO: add support for custom training etc
    learning_rate: int
    learning_rate_schedule : Optional[str] #TODO 

    def get_optimizer(self):
        if not custom:
            return eval("tf.keras.optimizers."+self.optimizer_type)(learning_rate = self.learning_rate)
        else:
            # To be filled for custom optimizers and learning rate schedules
            pass


# class LossFunctionSpec(BaseModel):
#    is_column_aggregation: bool
#    is_cell_aggregation: bool
#    metric_name: LossMetric
#    column_weights: Dict[str,float]

class Loss(BaseModel):
    custom: Optional[bool] = False
    loss_type: LossMetric
    # reduction: str # Options mean, 
    def get_loss(self):
    #TODO: add more support for custom losses, also add more support for params to loss functions
        if not custom:
            return eval("tf.keras.losses"+self.loss_type)()
        else:
            pass

class TimeSeriesTrainingConfig(TimeSeriesBaseConfig):
    
    search_space: dict
    search_parameters: dict
    train_loss_function: LossFunctionSpec
    optimizer: Optimizer
    loss: Loss
    epochs: int
    callbacks: list[str]
    batchsize: int

    def get_optimizer(self):
        return self.optimizer.get_optimizer()
    
    def get_loss(self):
        return self.loss.get_loss()

class TimeSeriesEvaluationConfig(TimeSeriesBaseConfig):
    pass

class TimeSeriesPredictionConfig(TimeSeriesBaseConfig):
    pass


class TabularBaseConfig(BaseModel):
    data_spec: TabularDataSpec
    model_class: ModelClass #TODO
    model_parameters: dict

class TabularTrainingConfig(TabularBaseConfig):
    search_space: dict
    search_parameters: dict
    train_loss_function: LossFunctionSpec
    optimizer: Optimizer
    loss: Loss
    epochs: int
    callbacks: list[str]
    batchsize: int
    def get_optimizer(self):
        self.optimizer.get_optimizer()
    
    def get_loss(self):
        self.loss.get_loss()