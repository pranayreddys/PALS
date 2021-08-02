from dataset.dataset import TimeSeriesDataset
from entities.key_entities import TimeSeriesDataSpec
from models.base_models import BaseTimeSeriesModel
import enum

"""
Refer folder simple_data_generation. Decided not to incorporate data generation for now
in the overall code structure, but some of the functions are implemented in simple_data_generation 
"""
@enum.unique
class Combination(str, enum.Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    MAXIMUM = "maximum"

@enum.unique
class EffectType(str, enum.Enum):
    ABSOLUTE = "absolute"
    DIFFERENCE = "difference"
    RATIO = "ratio"
    



class SimpleDelayedEffectModel(BaseTimeSeriesModel):

    def __init__(self,_dataspec: TimeSeriesDataSpec, _model_config):
        super(BaseTimeSeriesModel, self).__init__()
        self.dataspec = _dataspec
        self.stimulus_effect_lag = _model_config.stimulus_effect_lag
        self.effect_type = _model_config.effect_type
        self.combination = _model_config.combination


    def __predict_one_step(dependent_state_vals, independent_state_vals,control_input_vals):
        #TODO: need to implement this based on the model configuration
        return output

    
