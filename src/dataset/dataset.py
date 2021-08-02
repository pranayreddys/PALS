import pandas as pd
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec
from utils.read_write_utils import read_data
from sklearn.model_selection import train_test_split
from itertools import chain
from entities.all_enums import ExperimentMode
from typing import List
class TabularDataset:
    """
    Tabular Dataset class for interfacing with TabularData such as user profile params,
    for example personality type/motivation.
    TODO: This module has not been tested yet.
    """
    def __init__(self, _dataset_spec: TabularDataSpec, blank_dataset=False):
        self.dataset_spec =  _dataset_spec
        if not blank_dataset:
            self.data = read_data(dataset_spec.data_source, self.dataset_spec.data_format)
            self._validate()

    def _validate(self):
        """
        Currently performs very basic validation, just checks to see if columns exist
        """
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
        return self.data.loc[row_selection_query, column_selection_query]

class TimeSeriesDataset:
    def __init__(self, _dataset_spec: TimeSeriesDataSpec, blank_dataset=False):
        """
        TimeSeriesDataset class for interfacing with TimeSeriesData such as BP/UAT time series.
        """
        self.dataset_spec =  _dataset_spec
        self.data = pd.DataFrame()
        if not blank_dataset:
            self.data = read_data(_dataset_spec.data_source, _dataset_spec.data_format)
            self._validate()
            self._sort_values()

    def _check_existence(self, col_list):
        columns = set(self.data.columns)
        assert(set(col_list).issubset(columns))

    def _validate(self):
        columns = set(self.data.columns)
        self._check_existence(self.dataset_spec.independent_state_columns)
        self._check_existence(self.dataset_spec.control_input_columns)
        self._check_existence([self.dataset_spec.series_id_column])
        assert(self.dataset_spec.time_column in columns)
        self._check_existence(self.dataset_spec.series_attribute_columns)
        self._check_existence(self.dataset_spec.dependent_state_columns)
        assert(not (set(self.dataset_spec.dependent_state_columns) 
                & set(self.dataset_spec.independent_state_columns)))
        return

    def _sort_values(self):
        self.data.sort_values(self.dataset_spec.time_column, inplace=True)

    def describe(self):
         #TODO: Add additional time series information later  
         return self.data.describe()

    def detailed_report(self, report_path: str):
        """
        TODO: Refer Sharut's code and plot:
        a) individual column - availability, distribution
        b) plot original data, rolling mean and variance for day, week of each series x id (e.g., person's BP)
        c) plot auto-correlation
        d) run ADF-stationarity test, time series decomposition for each series x id
        e) plot mean and variance across series ids for each series; ADF stationarity test at population level  
        f) cross-correlations across different series
        """   
        return
    
    def select_subset(self, row_selection_query, column_selection_query):
        return self.data.loc[row_selection_query, column_selection_query]
    
    def subset_per_id(self):
        return self.data.groupby(self.dataset_spec.series_id_column) 
    
    def assign_id_vals(self, id, cols, values):
        self.data.loc[(self.data[self.dataset_spec.series_id_column]==id).values, cols] = values
    
    def _id_selection(self, ids):
        keys= ids.values.tolist()
        return pd.concat([self.data.loc[(self.data[self.dataset_spec.series_id_column] == key).values] for key in keys],
            ignore_index=True)
        

    def train_val_test_split(self, split_percentages: List[float], 
                        experiment_mode: ExperimentMode,
                        trainingconfig = None):
        """
        Helper function for creating train-val-test split.
        """
        train_dataset = TimeSeriesDataset(self.dataset_spec, blank_dataset=True)
        val_dataset= TimeSeriesDataset(self.dataset_spec, blank_dataset=True)
        test_dataset = TimeSeriesDataset(self.dataset_spec, blank_dataset=True)
        
        if experiment_mode==ExperimentMode.MultiTimeSeries:
            X= self.data[self.dataset_spec.series_id_column].drop_duplicates()
            X_train, X_test= train_test_split(X, test_size=split_percentages[2], random_state=1)
            X_train, X_val = train_test_split(X_train, 
                    test_size=split_percentages[1]/(split_percentages[0]+split_percentages[1]), random_state=1)
            train_dataset.data = self._id_selection(X_train)  
            train_dataset._sort_values()     
            val_dataset.data = self._id_selection(X_val)
            val_dataset._sort_values()
            test_dataset.data = self._id_selection(X_test)
            test_dataset._sort_values()
            return train_dataset, val_dataset, test_dataset
        elif experiment_mode==ExperimentMode.SimpleForecast:
            print("Mode not implemented, please implement this mode or use presplit datasets")
            raise NotImplementedError
            #TODO: Complete this training mode
            size = len(self.data)
            train_split_end = int(size*split_percentages[0])
            val_split_start = train_split_end-trainingconfig.context_window-trainingconfig.lead_gap
            assert val_split_start>0    
            val_split_end = int(size*split_percentages[1])
            test_split_start = val_split_end-trainingconfig.context_window-trainingconfig.lead_gap

