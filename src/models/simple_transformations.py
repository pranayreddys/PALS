from dataset.dataset import TabularDataset, TimeSeriesDataset
# from models.base_models import BaseTransformationModel
from sklearn import preprocessing
import pandas as pd
import enum
from dateutil import parser
from entities.all_enums import ColumnTransform
from typing import Union, Dict, List
import pickle
import os

class SimplePreprocessModel():
    """
    Performs simple transformations such as one hot encodings and other sklearn transforms including
    MinMaxScaling, StandardScaling etc.
    This class is utilized in base models as one of the sub components. 
    """
    def __init__(self, column_transformations: Dict[str, ColumnTransform] = None):
        self.column_transformations = column_transformations
        self.new_col_dict = {}
    
    def simple_fit_predict(self, dataset: Union[TabularDataset, TimeSeriesDataset], timeseries=True):
        if not self.column_transformations:
            return
        
        self._transformer= {}
        for col, transform in self.column_transformations.items():
            if transform not in [ColumnTransform.OneHotEncoder, ColumnTransform.DateTime, ColumnTransform.Identity]:
                self._transformer[col]= eval("preprocessing."+transform+"()")
                self._transformer[col].fit(dataset.data[col].values.reshape(-1,1))
        
        for col in dataset.data.columns:
            if col not in self.column_transformations.keys():
                self.column_transformations[col] = ColumnTransform.Identity
        
        ret_dataset = self.simple_predict(dataset, timeseries, training=True)
        self.dataspec = ret_dataset.dataset_spec
        return ret_dataset 

    def simple_predict(self, dataset: Union[TabularDataset, TimeSeriesDataset], timeseries=True, training=False):
        if not self.column_transformations:
            return
        if timeseries:
            ret_dataset = TimeSeriesDataset(dataset.dataset_spec, blank_dataset=True)
        else:
            ret_dataset = TabularDataset(dataset.dataset_spec, blank_dataset=True)
        
        for col, transform in self.column_transformations.items():
            if transform == ColumnTransform.OneHotEncoder:
                one_hot_encodings = pd.get_dummies(dataset.data[col]).astype('float')
                newcolumns = [col+"_"+str(onehotcol) for onehotcol in one_hot_encodings.columns]
                ret_dataset.data[newcolumns] = one_hot_encodings 
                if training:
                    self.new_col_dict[col] = set(newcolumns)
                else:
                    assert(set(newcolumns).issubset(self.new_col_dict[col]))
                    for column in self.new_col_dict[col]:
                        if column not in newcolumns:
                            ret_dataset.data[column] = 0
                ret_dataset.dataset_spec.replace(col, newcolumns)
                
            elif transform == ColumnTransform.DateTime:
                if col in dataset.data:
                    ret_dataset.data[col] = pd.to_datetime(dataset.data[col])
            elif transform == ColumnTransform.Identity:
                if col in dataset.data:
                    ret_dataset.data[col] = dataset.data[col]
            else:
                ret_dataset.data[col] = self._transformer[col].transform(dataset.data[col].values.reshape(-1,1)).squeeze()
        return ret_dataset
    
    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump([self._transformer, self.new_col_dict, self.column_transformations, self.dataspec], f)
    
    def load(self,path):
        with open(path, 'rb') as f:
            self._transformer, self.new_col_dict, self.column_transformations, self.dataspec = pickle.load(f)
        
        return self.dataspec
    
    def invert(self, predictions, predicted_columns: List[str], mapping: Dict[str, str]):
        for i,col in enumerate(predicted_columns):
            if self.column_transformations[mapping[col]] not in [ColumnTransform.DateTime, ColumnTransform.Identity, ColumnTransform.OneHotEncoder]:
                predictions[:, i] = self._transformer[mapping[col]].inverse_transform(predictions[:, i])
        
        return predictions



# class SimpleTransformationsModel(BaseTransformationModel):
# #TODO: when required
#     def __init__(self,_dataspec: TabularDataSpec, _model_config):
#         super(BaseTransformationModel, self).__init__()
#         self.dataspec = _dataspec
#         #TODO: shall we assume _model_config is a list of transformations 
#         # to be applied to individual input columns - some classes already exist
        
#     def _simple
#     def _predict(self,independent_vals):

#         pass
#       #TODO: need to predict the transformation
# class Normalize(BaseTransformationModel):
    
#     def _predict(self, independent_vals):

