from dataset.dataset import TabularDataset, TimeSeriesDataset
from models.base_models import BaseTransformationModel
from sklearn import preprocessing
import pandas as pd
import enum
from dateutil import parser
from entities.all_enums import ColumnTransform
from typing import Union, Dict
class SimplePreprocessModel():
    def __init__(self, column_transformations: Dict[str, ColumnTransform]):
        self.column_transformations = column_transformations
    
    def simple_fit(self, dataset: Union[TabularDataset, TimeSeriesDataset]):
        if not self.column_transformations:
            return
        
        self._transformer= {}
        for col, transform in self.column_transformations.items():
            if transform not in [ColumnTransform.OneHotEncoder, ColumnTransform.DateTime]:
                self._transformer[col]= eval("preprocessing."+transform+"()")
                self._transformer[col].fit(dataset.data[col].values.reshape(-1,1))
        
        for col in dataset.data.columns:
            if col not in self.column_transformations.keys():
                self.column_transformations[col] = ColumnTransform.Identity

    def simple_predict(self, dataset, timeseries=True):
        if not self.column_transformations:
            return
        if timeseries:
            ret_dataset = TimeSeriesDataset(dataset.dataset_spec, blank_dataset=True)
        else:
            ret_dataset = TabularDataset(dataset.dataset_spec, blank_dataset=True)
        
        for col, transform in self.column_transformations.items():
            if transform == ColumnTransform.OneHotEncoder:
                one_hot_encodings = pd.get_dummies(dataset.data[col])
                ret_dataset.data[one_hot_encodings.columns] = one_hot_encodings 
            elif transform == ColumnTransform.DateTime:
                ret_dataset.data[col] = pd.to_datetime(dataset.data[col])
            elif transform == ColumnTransform.Identity:
                ret_dataset.data[col] = dataset.data[col]
            else:
                ret_dataset.data[col] = self._transformer[col].transform(dataset.data[col].values.reshape(-1,1)).squeeze()
        
        return ret_dataset

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

