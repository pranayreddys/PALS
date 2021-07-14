import pandas as pd
from entities.key_entities import TabularDataSpec, TimeSeriesDataSpec
from utils.read_write_utils import read_data
from sklearn.model_selection import train_test_split
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
        self.data = pd.DataFrame()
        if not blank_dataset:
            self.data = read_data(_dataset_spec.data_source, _dataset_spec.data_format)
            self._validate()
            self.data.sort_values(self.dataset_spec.time_column, inplace=True)
        # validate data with the schema 

    def _check_existence(self, col_list):
        columns = set(self.data.columns)
        assert(set(col_list).issubset(columns))

    def _validate(self):
        #TODO
        columns = set(self.data.columns)
        self._check_existence(self.dataset_spec.independent_state_columns)
        self._check_existence(self.dataset_spec.control_input_columns)
        self._check_existence(self.dataset_spec.series_id_columns)
        assert(self.dataset_spec.time_column in columns)
        self._check_existence(self.dataset_spec.series_attribute_columns)
        self._check_existence(self.dataset_spec.dependent_state_columns)
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
    
    def assign_id_vals(self, id, cols, values):
        self.data.loc[self.data[self.dataset_spec.series_id_columns]==id, cols] = values
    
    def _id_selection(self, ids):
        keys = list(ids)
        i1 = self.data.set_index(keys).index
        i2  = ids.set_index(keys).index
        return self.data[i1.isin(i2)]

    def train_val_test_split(self, split_percentages):
        X= self.data[self.dataset_spec.series_id_columns].drop_duplicates()
        X_train, X_test= train_test_split(X, test_size=split_percentages[2], random_state=1)
        X_train, X_val = train_test_split(X_train, 
                test_size=split_percentages[1]/(split_percentages[0]+split_percentages[1]), random_state=1)
        train_dataset = TimeSeriesDataset(self.dataset_spec, blank_dataset=True)
        train_dataset.data = self._id_selection(X_train)        
        val_dataset= TimeSeriesDataset(self.dataset_spec, blank_dataset=True)
        val_dataset.data = self._id_selection(X_val)
        test_dataset = TimeSeriesDataset(self.dataset_spec, blank_dataset=True)
        test_dataset.data = self._id_selection(X_test)
        return train_dataset, val_dataset, test_dataset

