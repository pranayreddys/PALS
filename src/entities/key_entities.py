from typing import List, Dict, NewType
from pydantic import BaseModel
from entities.all_enums import TimeUnit, LossMetric, Distribution

FilePath = NewType("FilePath", str)
class TimeSeriesDataSpec(BaseModel):
    
    #TODO: Pranay please fix this appropriately as per the AutoML input
    data_source: FilePath
    data_format: str
    data_granularity_unit: TimeUnit # How are we going to use these? Imputation in the code?
    data_granularity_count: int # We use this for sampling?
    
    #CHECK: keeping these as lists in case the information is spread across multiple columns
    #Feel free to change time and series_id to a single one
    time_column: str
    series_id_columns: List[str]
    series_attribute_columns: List[str]
    control_input_columns: List[str] 
    dependent_state_columns: List[str] 
    independent_state_columns: List[str]

    #TODO: Add validation everywhere - in this case, some of it can happen here some in the dataset construction
    @classmethod
    def validate(cls, v):
    	return v
      
class TabularDataSpec(BaseModel):
   #TODO: This is to capture static profile of individuals 
   data_source: FilePath
   data_format: str
   instance_id_columns: List[str]
   dependent_columns: List[str] 
   independent_columns: List[str]
   target_columns: List[str]
   
class LossFunctionSpec(BaseModel):
   is_column_aggregation: bool
   is_cell_aggregation: bool
   metric_name: LossMetric
   column_weights: Dict[str,float]
   #TODO: need additional elements for counterfactual evaluation 
   
class VariableSpec(BaseModel):
#TODO: just the information needed to sample from tfp.distributions.Distribution
   variable_name:str
   distribution_name: Distribution
   distribution_args: Dict[str,str]
  
