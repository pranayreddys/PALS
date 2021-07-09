import pandas as pd
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec
from utils.read_write_utils import read_data
class TabularDataset:
	def __init__(self, _dataset_spec: TabularDataSpec, blank_dataset=False):
		#TODO
		# read the dataset from data source
		# self.data
		# set the schema
		self.dataset_spec =  _dataset_spec
		if not blank_dataset:
			self.data = read_data(dataset_spec.data_source, self.dataset_spec.data_format)
			self._validate()
		# validate data with the schema 

	def _validate(self):
		#FIXME: If more validation required
		columns = set(self.data.columns)
		assert(set(self.dataset_spec.instance_id_columns).issubset(columns))
		assert(set(self.dataset_spec.dependent_columns).issubset(columns))
		assert(set(self.dataset_spec.independent_columns).issubset(columns))
		assert(not (set(self.dataset_spec.dependent_columns) 
				& set(self.dataset_spec.independent_columns)))
		
		return

 	def describe(self):
 		return self.data.describe()

 	def detailed_report(self, report_path: str):
 		#TODO-SKIP FOR NOW - NOT NEEDED
    		#individual column - availability, distribution 
    		#pairwise correlations
 		return
	def select_subset(self, row_selection_query, column_selection_query):
		# TODO: use the selection_query to get a subset of rows and a list of columns
		return self.data.loc[row_selection_query, column_selection_query]

class TimeSeriesDataset:
	def __init__(self, _dataset_spec: TimeSeriesDataSpec, blank_dataset=False):
		#TODO
		# read the dataset from data source
		# self.data 
		# set the schema
		self.dataset_spec =  _dataset_spec
		if not blank_dataset:
			self.data = read_data(dataset_spec.data_source, dataset_spec.data_format)
			self._validate()
			self.data.sort_values(self.dataset_spec.time_column, inplace=True)
		# validate data with the schema 

	def _validate(self):
		#TODO
		columns = set(self.data.columns)
		assert(set(self.dataset_spec.independent_state_columns).issubset(columns))
		assert(set(self.dataset_spec.control_input_columns).issubset(columns))
		assert(self.dataset_spec.time_column in columns)
		assert(self.dataset_spec.series_id_column in columns)
		assert(self.dataset_spec.series_attribute_columns in columns) # TODO: Is this required? 
		assert(self.dataset_spec.dependent_state_columns in columns)
		assert(not (set(self.dataset_spec.dependent_state_columns) 
				& set(self.dataset_spec.independent_state_columns)))
		return

 	def describe(self):
 		#TODO: Add additional time series information later  
 		return self.data.describe()

 	def detailed_report(self, report_path: str):
 		# TODO - LATER --statsmodels - we can use Sharut's code
   		# individual column - availability, distribution
 		# plot original data, rolling mean and variance for day, week of each series x id (e.g., person's BP)
 		# plot auto-correlation
 		# run ADF-stationarity test, time series decomposition for each series x id
 		# plot mean and variance across series ids for each series; ADF stationarity test at population level  
 		# cross-correlations across different series
 		return
	
	def select_subset(self, row_selection_query, column_selection_query):
		# TODO: use the selection_query to get a subset of rows and a list of columns
		return self.data.loc[row_selection_query, column_selection_query]
	
	def subset_per_id(self):
		return self.data.groupby(self.dataset_spec.series_id_columns) 
	
	def assign_id(self, id, cols, values):
		self.data.loc[self.data[self.dataset_spec.series_id_columns]==id, cols] = values
