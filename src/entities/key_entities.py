from typing import List, Dict, NewType
from pydantic import BaseModel
from entities.all_enums import TimeUnit, LossMetric, DistributionType, ColumnTransform

class TimeSeriesDataSpec(BaseModel):
    
      #TODO: Pranay please fix this appropriately as per the AutoML input
      # Support only for a single key
   data_source: str
   data_format: str
   data_granularity_unit: TimeUnit = TimeUnit.Day # How are we going to use these? Imputation in the code?
   data_granularity_count: int = 1 # We use this for sampling?

   #CHECK: keeping these as lists in case the information is spread across multiple columns
   #Feel free to change time and series_id to a single one
   time_column: str
   series_id_column: str
   series_attribute_columns: List[str]
   control_input_columns: List[str] 
   dependent_state_columns: List[str] 
   independent_state_columns: List[str]
   #TODO: Add validation everywhere - in this case, some of it can happen here some in the dataset construction
   def replace(self, col, newcolumns):
      for column_list in [self.series_attribute_columns, self.control_input_columns, 
            self.dependent_state_columns, self.independent_state_columns]:
         if col in column_list:
            column_list.remove(col)
            column_list.append(newcolumns)



class TabularDataSpec(BaseModel):
   #TODO: This is to capture static profile of individuals 
   data_source: str
   data_format: str
   instance_id_columns: List[str]
   dependent_columns: List[str] 
   independent_columns: List[str]
   target_columns: List[str]
   
   def delete(self, columns):
      setcols = set(columns)
      setinstance_ids = set(self.instance_id_columns)
      setdep_cols = set(self.dependent_columns)
      setindep_cols = set(self.independent_columns)
      settarget_cols = set(self.target_columns)
      self.instance_id_columns = list(setinstance_ids.difference(setcols))
      self.dependent_columns = list(setdep_cols.difference(setcols))
      self.independent_columns = list(setindep_cols.difference(setcols))
      self.target_columns = list(settarget_cols.difference(setcols))
   
   def replace(self, col, newcolumns):
      for column_list in [self.instance_id_columns, self.dependent_columns, 
            self.independent_columns, self.target_columns]:
         if col in column_list:
            column_list.remove(col)
            column_list.append(newcolumns)
      

   #TODO: need additional elements for counterfactual evaluation 
   
class VariableSpec(BaseModel):
   variable_name:str 
   distribution_name: DistributionType
   distribution_args: Dict[str, float]
   def init_distribution(self):
      self.distribution_object = eval("tfp.distributions."+self.distribution_type)(
                                       **self.distribution_params)

   def sample_distribution(self, shape=(1,)):
      return self.distribution_object.sample(shape)