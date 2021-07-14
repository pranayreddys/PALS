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

class Runner(BaseModel):
    training: bool = True
    dataset_spec: TimeSeriesDataSpec
    presplit: bool = False
    mode: ExperimentMode = ExperimentMode.MultiTimeSeries
    test_output: str
    split_percentages: Optional[List[float]] 
    train_path: Optional[str]
    val_path: Optional[str]
    test_path: Optional[str]
    training_config: Optional[TimeSeriesTrainingConfig]
    eval_config: Optional[TimeSeriesEvaluationConfig] = None
    predict_config: Optional[TimeSeriesPredictionConfig] = None
    model_checkpoint: Optional[str]
    training_stats: Optional[str]=None
    evaluation_stats: Optional[str]=None

    #TODO: write validators to check existence of files above
    #TODO: write validators to ensure contradictory inputs not provided.
    def _validate(self):
        assert(self.presplit == (self.split_percentages==None))
        if self.presplit:
            for path in [self.train_path, self.val_path, self.test_path]:
                is_file(path)
        if self.training:
            assert(self.training_config)
            if not self.eval_config:
                self.eval_config = self.training_config
            if not self.predict_config:
                self.predict_config = self.training_config

            
            
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
            
            #Preprocess the data: TODO Move to one big model
            preprocessor = SimplePreprocessModel(self.dataset_spec.column_transformations)
            preprocessor.simple_fit(train_dataset)
            train_dataset  = preprocessor.simple_predict(train_dataset)
            val_dataset = preprocessor.simple_predict(val_dataset)
            test_dataset = preprocessor.simple_predict(test_dataset)
            ###

            model = get_model(self.training_config)(train_dataset.dataset_spec)
            model.set_params(self.training_config.model_parameters)
            #TODO: Model checkpoints, fit only when required
            history = model.simple_fit(train_dataset, val_dataset, self.training_config)
            with open(self.training_stats, 'w') as f:
                f.write(tabulate.tabulate(history.history, headers=history.history.keys()))
            prediction = model.simple_predict(test_dataset, self.predict_config)
            prediction.data.to_csv(self.test_output)
            eval_results = model.simple_evaluate(val_dataset, self.predict_config)
            with open(self.evaluation_stats, 'w') as f:
                f.write(tabulate.tabulate(eval_results, headers=history.history.keys()))

        else:
            pass
