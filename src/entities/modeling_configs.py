from pydantic import BaseModel
from typing import Optional, Dict, List
from entities.key_entities import TimeSeriesDataSpec, TabularDataSpec
from entities.all_enums import OptimizerType, LossMetric, DistributionType, ModelClass
import tensorflow as tf
import tensorflow_probability as tfp
# from models.bp_models import *
# from models.base_models import *


class TimeSeriesBaseConfig(BaseModel):
    # data_spec: TimeSeriesDataSpec Dataspec kept separate from baseconf
    model_class: ModelClass
    model_parameters: Dict
    context_window: int 
    lead_gap: int 
    forecast_horizon: int 
    stride: int = 1
    output_file_prefix: str = None
    output_dir: str = None

class Optimizer(BaseModel):
    custom: Optional[bool] = False
    optimizer_type : OptimizerType
    # TODO: add support for custom training etc
    learning_rate: int = 0.1
    learning_rate_schedule : Optional[str] #TODO 

    def get_optimizer(self):
        if not self.custom:
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
    column_weights: Optional[Dict[str, float]] = None
    is_column_aggregation: bool = True
    is_cell_aggregation: bool = True
    # reduction: str # Options mean, 
    def get_loss(self):
    #TODO: add more support for custom losses, also add more support for params to loss functions
        if not self.custom:
            return eval("tf.keras.losses."+self.loss_type)()
        else:
            pass
        


class TimeSeriesTrainingConfig(TimeSeriesBaseConfig):
    search_space: Optional[Dict]
    search_parameters: Optional[Dict]
    optimizer: Optimizer
    train_loss_function: Loss
    epochs: int
    metrics: List[LossMetric] = None #TODO
    batchsize: int

    def get_optimizer(self):
        return self.optimizer.get_optimizer()
    
    def get_loss(self):
        return self.train_loss_function.get_loss()
    
    def get_metrics(self):
        return [eval("tf.keras.losses."+loss)() for loss in self.metrics]

class TimeSeriesEvaluationConfig(TimeSeriesBaseConfig):
    loss_list: List[LossMetric]
    def get_losses(self):
        return [("tf.keras.losses."+loss)() for loss in self.loss_list]
    

class TimeSeriesPredictionConfig(TimeSeriesBaseConfig):
    pass

class TimeSeriesImputationConfig(TimeSeriesBaseConfig):
    pass

class TimeSeriesOutlierDetectionConfig(TimeSeriesBaseConfig):
    pass

class TimeSeriesControlConfig(TimeSeriesBaseConfig):
    pass

class TabularBaseConfig(BaseModel):
    data_spec: TabularDataSpec
    model_class: ModelClass #TODO
    model_parameters: Dict

class TabularTrainingConfig(TabularBaseConfig):
    search_space: Dict
    search_parameters: Dict
    optimizer: Optimizer
    train_loss_function: Loss
    epochs: int
    callbacks: list[str]
    batchsize: int
    def get_optimizer(self):
        self.optimizer.get_optimizer()
    
    def get_loss(self):
        self.loss.get_loss()

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