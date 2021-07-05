import pandas as pd
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec

class TabularDataset(pd.DataFrame):

	def _init_(self, _dataset_spec:TabularDataSpec):
		#TODO
		# read the dataset from data source
		# self.data
		# set the schema
		dataset_spec =  _dataset_spec
		# validate data with the schema 

	def _validate(self):
		#TODO
		return

 	def describe(self):
 		return self.data.describe()

 	def detailed_report(self, report_path:str):
 		#TODO-SKIP FOR NOW - NOT NEEDED
    		#individual column - availability, distribution 
    		#pairwise correlations
 		return
	def select_subset(self, row_selection_query, column_selection_query):
		# TODO: use the selection_query to get a subset of rows and a list of columns
		return df_subset

class TimeSeriesDataset(pd.DataFrame):

	def _init_(self, _dataset_spec:TimeSeriesDataSpec):
		#TODO
		# read the dataset from data source
		# self.data 
		# set the schema
		dataset_spec =  _dataset_spec
		# validate data with the schema 

	def _validate(self):
		#TODO
		return

 	def describe(self):
 		#TODO: Add additional time series information later  
 		return self.data.describe()

 	def detailed_report(self, report_path:str):
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
		return df_subset
