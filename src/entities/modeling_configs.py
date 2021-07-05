from pydantic import BaseModel
from entities.key_entities import TimeSeriesDataSpec, TabularDataSpec, LossFunctionSpec


class TimeSeriesBaseConfig(BaseModel):
    data_spec: TimeSeriesDataSpec
    model_class: ModelClass # TODO: need to create this
    model_parameters: dict
    fixed_lead: bool 
    fixed_anchor_index: bool 
    lead_gap: int
    anchor_index: str # or DateTime?
    start_index: str
    end_index: str
    output_file_prefix: str = None
    output_dir: str = None
    

class TimeSeriesTrainingConfig(TimeSeriesBaseConfig):
 
    search_space: dict
    search_parameters: dict
    train_loss_function: LossFunctionSpec
