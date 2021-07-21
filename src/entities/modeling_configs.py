from pydantic import BaseModel
from typing import Optional, Dict, List
from entities.key_entities import TimeSeriesDataSpec, TabularDataSpec
from entities.all_enums import OptimizerType, LossMetric, DistributionType, ModelClass, ColumnTransform
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
    model_save_folder: Optional[str]

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
            print("Custom optimizers not implemented")
            raise NotImplementedError



class Loss(BaseModel):
    custom: Optional[bool] = False
    loss_type: Optional[LossMetric] = None
    column_weights: Optional[Dict[str, float]] = None
    column_loss_type: Optional[Dict[str, LossMetric]] = None
    is_column_aggregation: bool = True
    is_cell_aggregation: bool = True
    # reduction: str # Options mean, 
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
    search_space: Optional[Dict]
    search_parameters: Optional[Dict]
    optimizer: Optimizer
    train_loss_function: Loss
    epochs: int
    metrics: List[LossMetric] = None #TODO
    batchsize: int
    column_transformations: Dict[str, ColumnTransform] = None

    def get_optimizer(self):
        return self.optimizer.get_optimizer()
    
    def get_loss(self, column_order=None):
        return self.train_loss_function.get_loss(column_order)
    
    def get_metrics(self):
        return [eval("tf.keras.losses."+loss)() for loss in self.metrics]

class TimeSeriesEvaluationConfig(TimeSeriesBaseConfig):
    #TODO: use this config in simple_evaluate
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
    column_transformations: Dict[str, ColumnTransform] = None

class TabularTrainingConfig(TabularBaseConfig):
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