from pydantic import BaseModel, parse_file_as
from typing import Dict, Optional, List
from entities.key_entities import TimeSeriesDataSpec
from entities.modeling_configs import TimeSeriesTrainingConfig, \
        TimeSeriesEvaluationConfig, TimeSeriesPredictionConfig

from dataset.dataset import TimeSeriesDataset, TabularDataset
from utils.read_write_utils import is_file
from models.simple_transformations import SimplePreprocessModel
from models.models import SimpleVAR
from entities.all_enums import ExperimentMode
from models.models import get_model
import tabulate
from utils.eda import visualize
from publishers.mlflow_logging import log_to_mlflow
class Runner(BaseModel):
    training: bool = True
    dataset_spec: TimeSeriesDataSpec
    presplit: bool = False
    mode: ExperimentMode = ExperimentMode.MultiTimeSeries
    test_output: Optional[str]=None
    split_percentages: Optional[List[float]] 
    train_path: Optional[str]
    val_path: Optional[str]
    test_path: Optional[str]
    training_config: Optional[TimeSeriesTrainingConfig]
    eval_config: Optional[TimeSeriesEvaluationConfig] = None
    predict_config: Optional[TimeSeriesPredictionConfig] = None
    model_checkpoint: Optional[str]

    def _validate(self):
        assert(self.presplit == (self.split_percentages==None))
        if self.training:
            if self.presplit:
                for path in [self.train_path, self.val_path, self.test_path]:
                    is_file(path)
            assert(self.training_config)
            if not self.eval_config:
                self.eval_config = self.training_config
            if not self.predict_config:
                self.predict_config = self.training_config
            assert(self.training_config.model_save_folder)
        else:
            is_file(self.test_path)
            assert(self.training_config)
            if not self.predict_config:
                self.predict_config = self.training_config
            assert(self.presplit)
            assert(self.predict_config.model_save_folder)


            
            
    def run(self):
        #TODO: Training, eval, test splitting? 
        #group by id and assign different IDs for train,eval and test.
        self._validate()
        if self.training:
            ## Splitting into train-val-test
            if not self.presplit:
                full_dataset = TimeSeriesDataset(_dataset_spec=self.dataset_spec)
                train_dataset, val_dataset, test_dataset= \
                         full_dataset.train_val_test_split(self.split_percentages, self.mode, self.training_config)
            else:
                self.dataset_spec.data_source = self.train_path
                train_dataset = TimeSeriesDataset(_dataset_spec=self.dataset_spec)
                self.dataset_spec.data_source = self.val_path
                val_dataset = TimeSeriesDataset(_dataset_spec=self.dataset_spec)
                self.dataset_spec.data_source = self.test_path
                test_dataset = TimeSeriesDataset(_dataset_spec=self.dataset_spec)
            model = get_model(self.training_config)(train_dataset.dataset_spec)
            model.set_params(self.training_config.model_parameters)
            history = model.simple_fit(train_dataset, val_dataset, self.training_config)
            artifacts = {}
            model.save_model(self.training_config.model_save_folder)
            artifacts["model_path"] = self.training_config.model_save_folder
            if self.test_output:
                prediction = model.simple_predict(test_dataset, self.predict_config)
                prediction.data.to_csv(self.test_output)
                artifacts["test_output"] = self.test_output
            eval_results = model.simple_evaluate(test_dataset, self.predict_config)
            log_to_mlflow(self.dict(), history.history|eval_results, artifacts)
        else:
            self.dataset_spec.data_source = self.test_path
            test_dataset = TimeSeriesDataset(_dataset_spec=self.dataset_spec)
            model = get_model(self.training_config)(self.dataset_spec)
            model.set_params(self.training_config.model_parameters)
            model.load_model(self.training_config.model_save_folder, self.training_config)
            artifacts = {}
            if self.test_output:
                prediction = model.simple_predict(test_dataset, self.predict_config)
                prediction.data.to_csv(self.test_output)
                artifacts["test_output"] = self.test_output
            
            eval_results = model.simple_evaluate(test_dataset, self.predict_config)
            eval_results= {("eval_"+id+"_"+k):value for k,value in eval_dict.items() for id,eval_dict in eval_results}
            log_to_mlflow(self.dict(), eval_results, artifacts)
