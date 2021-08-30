from typing import List, Dict, NewType, Optional
from pydantic import BaseModel
from entities.all_enums import TimeUnit, LossMetric, DistributionType, ColumnTransform

class TimeSeriesDataSpec(BaseModel):
   """
   This class is used to describe the Data Spec for the time series data provided.
   Note that certain features like data_granularity_unit and data_granularity_count are not implemented yet.
   Data format currently supported is only csv.
   Args:
      data_source - file from which to read data
      data_format - type of file (example csv). Currently only csv implemented
      data_granularity_unit - Parameter to define granularity, not used right now.
      data_granularity_count - Parameter to define number of granularity units, not used right now.
      time_column - Specify the time column of the time series, it will be sorted according to this column
      series_id_column - Specify the series ID for the time series (example User ID).
      series_attribute_columns - Specify attribute columns in multi-time series data. For example user age/ user
      psychological profile.
      control_input_columns - Specify control input columns like nudge input that is provided.
      dependent_state_columns - Specify the dependent state columns (e.g. UAT)
      independent_state_columns - Specify independent state columns (e.g. BP)
   """
   data_source: str
   data_format: str
   data_granularity_unit: TimeUnit = TimeUnit.Day # TODO: Not used right now
   data_granularity_count: int = 1 # TODO: Not used right now, current code expects data to have no missing values

   time_column: str
   series_id_column: str
   series_attribute_columns: List[str]
   control_input_columns: List[str] 
   dependent_state_columns: List[str] 
   independent_state_columns: List[str]
   def replace(self, col, newcolumns):
      """
      Helper function that is used while performing column transformations. After column transformations,
      the spec changes at times (for example when one-hot encoding). This function handles that case.
      """
      for column_list in [self.series_attribute_columns, self.control_input_columns, 
            self.dependent_state_columns, self.independent_state_columns]:
         if col in column_list:
            column_list.remove(col)
            if isinstance(newcolumns, list):
               column_list.extend(newcolumns)
            else:
               column_list.append(newcolumns)



class TabularDataSpec(BaseModel):
   """
   This is for modelling and utilizing user static profile.
   TODO: Currently untested and unused - might be useful when user static variables are required.
   """
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
      """
      Function that is utilized for column transformations. Some column transformations (such as OneHotEncoding)
      change the 
      """
      for column_list in [self.instance_id_columns, self.dependent_columns, 
            self.independent_columns, self.target_columns]:
         if col in column_list:
            column_list.remove(col)
            if isinstance(newcolumns, list):
               column_list.extend(newcolumns)
            else:
               column_list.append(newcolumns)
            
      
class NudgeOptimizationDataSpec(BaseModel):
   """Dataspec for the nudge optimization part of the code. The assumption is that data consists of tuples
   (X, y, \delta) where X is the state, y is the action and \delta is the reward for that nudge.
   In our case, X is basically an unwrapped form of the timeseries data, y is the nudge, and \delta is a reward
   like BP reduction. 
   
   Args:
      data_source : The path to the data
      data_format : Format of the data at the path (e.g. csv). Currently only csv has been implemented.
      time_column : Optional, the name of the column representing time. Added as a parameter just in case it is required for
                     certain models.
      series_id_column : Again optional, if the model requires the series ID then it needs to be provided. At present,
                        none of the nudge optimizer and reward predictor models make use of series ID, hence it is not used.
      action_column : Name of the column which has the actions. Note that actions themselves are assumed to be integers in the range [0, num_actions-1]
      reward_column : Name of the csv column that represents observed reward for the state action tuple
      state_columns :  Columns that encapsulate the state. Currently there are assumptions on this format for the time series reward predictor and time series nudge optimizer.
      sampling_policy: List of columns with the probability of suggesting action | state. There are num_action such columns. The order of these columns needs to be 
                     of the form [probability_0, probability_1, ... probability_[num_action-1]] in the config (probability_actionnum in ascending order)

   """
   data_source: str
   data_format: str
   time_column: Optional[str] = None
   series_id_column: Optional[str] = None
   action_column: str
   reward_column: str
   state_columns: List[str]
   sampling_policy: List[str]

   
class VariableSpec(BaseModel):
   """
   Defines and samples from distribution objects. Currently not utilized in code.
   Instead refer to simple_data_generation folder, contains data generation pipeline that can be
   utilized for generating data according to certain assumptions.
   """
   variable_name:str 
   distribution_name: DistributionType
   distribution_args: Dict[str, float]
   def init_distribution(self):
      #TODO: Move from tensorflow_probability to numpy. On GCP there are problems with tensorflow_probability
      self.distribution_object = eval("tfp.distributions."+self.distribution_type)(
                                       **self.distribution_params)

   def sample_distribution(self, shape=(1,)):
      return self.distribution_object.sample(shape)