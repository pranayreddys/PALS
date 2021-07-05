from typing import List
from pydantic import BaseModel

class TimeSeriesDataSpec(BaseModel):
    
    #TODO: Pranay please fix this appropriately as per the AutoML input
    data_source: FilePath
    data_granularity_unit: TimeUnit
    data_granularity_count: int
    #CHECK: keeping these as lists in case the information is spread across multiple columns
    #Feel free to change time and series_id to a single one
    time_columns: List[str]
    series_id_columns: List[str]
    series_attribute_columns: List[str]
    control_input_columns: List[str] 
    dependent_state_columns: List[str] 
    independent_state_columns: List[str]

    #TODO: Add validation everywhere - in this case, some of it can happen here some in the dataset construction
    def validate(cls, v):
    	return v
      
 class TabularDataSpec(BaseModel):
    
    #TODO: This is to capture static profile of individuals 
    data_source: FilePath
    instance_id_columns: List[str]
    dependent_columns: List[str] 
    independent_columns: List[str]
      
 class LossFunctionSpec(BaseModel):
    is_column_aggregation: bool
    is_cell_aggregation: bool
	  metric_name: LossMetric
	  column_weights: Dict[str,float]
    
 class VariableSpec(BaseModel):
   #TODO: just the information needed to sample from tfp.distributions.Distribution
   variable_name:str
   distribution_name: Distribution
   distribution_args: Dict[str,str]
  
