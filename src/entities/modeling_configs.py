from pydantic import BaseModel
from entities.key_entities import TimeSeriesDataSpec, TabularDataSpec, LossFunctionSpec
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
    # fixed_lead: bool 
    # fixed_anchor_index: bool
    # anchor_index: str # or DateTime?
    # start_index: str
    # end_index: str
    
    

class Optimizer(BaseModel):
    optimizer_type : str
    learning_rate: int
    learning_rate_schedule : str

    def get_optimizer(self):
        #TODO: Construct optimizer object
        optimizers = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD
            }
        return optimizers[self.optimizer_type](learning_rate = self.learning_rate)

class TimeSeriesTrainingConfig(TimeSeriesBaseConfig):
    
    search_space: dict
    search_parameters: dict
    train_loss_function: LossFunctionSpec
    optimizer: Optimizer
    epochs: int
    callbacks: list[str]
    batchsize: int

    def get_optimizer(self):
        return self.optimizer.get_optimizer()

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
    epochs: int
    callbacks: list[str]
    batchsize: int
    def get_optimizer(self):
        self.optimizer.get_optimizer()