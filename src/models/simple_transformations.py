from dataset.dataset import TabularDataset
from models.base_models import BaseTransformationModel
import enum

class SimpleTransformationsModel(BaseTransformationModel):

    def __init__(self,_dataspec: TabularDataSpec, _model_config):
        super(BaseTransformationModel, self).__init__()
        self.dataspec = _dataspec
        #TODO: shall we assume _model_config is a list of transformations 
        # to be applied to individual input columns - some classes already exist
        
    @staticmethod
    def _predict(independent_vals):
        pass
      #TODO: need to predict the transformation
