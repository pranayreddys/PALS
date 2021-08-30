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
from models.models import get_forecasting_model
import tabulate
from utils.eda import visualize
from publishers.mlflow_logging import log_to_mlflow

class Runner(BaseModel):
    """
    Basic runner code, needs to be improved. Currently performs training for a time series model given data
    Args:
        training : Mode that defines whether the model is to be trained or a model needs to be loaded for prediction
        dataset_spec : Spec such as Time Column, Series Attribute columns etc. Refer :class:`~src.entities.key_entitites.TimeSeriesDataSpec`
        presplit : If False, splits dataset according to mode. If True, need to provide train_path, val_path and test_path for train, val,test splits.
        mode : Refer :class:`~src.entities.all_enums.ExperimentMode` for possible modes and meaning. The splitting logic is implemented in :func:`~src.dataset.dataset.TimeSeriesDataset.train_val_test_split` for modes.
        test_output: Dumps test output into a csv at this path
        split_percentages: Used when presplit is False, then split_percentages need to be provided (list of 3 floats from 0 to 1 representing train, val, test)
        train_path : Used when presplit is False 
        val_path : Used when presplit is False
        test_path : Used when presplit is False
        training_config : Has configuration params such as context_window, forecast horizon. Needs to be provided for both model loading (same config used during training) and for training purposes
        eval_config : Not implemented yet, but might be useful for other forms of evaluation.
        predict_config : Used during prediction. If this is not provided then prediction_config params would default to training_config params
        experiment_name : Name of experiment to be provided to mlflow
        run_name : Name of run to be provided to mlflow
        tags : Tags to be provided to mlflow
        train_output : Optional, dump the predictions of train into this path (if present)
    """
    training: bool = True
    dataset_spec: TimeSeriesDataSpec
    presplit: bool = False
    mode: ExperimentMode = ExperimentMode.MultiTimeSeries
    test_output: Optional[str]=None
    split_percentages: Optional[List[float]] 
    train_path: Optional[str]
    val_path: Optional[str]
    test_path: Optional[str]
    training_config: TimeSeriesTrainingConfig
    eval_config: Optional[TimeSeriesEvaluationConfig] = None
    predict_config: Optional[TimeSeriesPredictionConfig] = None
    experiment_name: Optional[str] = "Default"
    run_name: Optional[str] = None
    tags : Optional[Dict[str, str]] = None
    train_output : Optional[str] = None

    def _validate(self):
        assert(self.presplit == (self.split_percentages==None))
        if self.training:
            if self.presplit:
                for path in [self.train_path, self.val_path, self.test_path]:
                    is_file(path)
            assert self.training_config
            if not self.eval_config:
                self.eval_config = self.training_config
            if not self.predict_config:
                self.predict_config = self.training_config
            assert self.training_config.model_save_folder
        else:
            is_file(self.test_path)
            assert self.training_config
            if not self.predict_config:
                self.predict_config = self.training_config
            assert self.presplit
            assert self.predict_config.model_save_folder


            
    def run(self):
        self._validate()
        if self.training:
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
            model = get_forecasting_model(self.training_config.model_class)(train_dataset.dataset_spec)
            history = model.simple_fit(train_dataset, val_dataset, self.training_config, self.training_config.model_parameters)
            artifacts = {}
            if self.train_output:
                model.simple_predict(train_dataset, self.training_config).data.to_csv(self.train_output)
                artifacts["train_output"]= self.train_output
            model.save_model(self.training_config.model_save_folder)
            model.visualize(self.training_config.model_save_folder)
            artifacts["model_path"] = self.training_config.model_save_folder
            if self.test_output:
                prediction = model.simple_predict(test_dataset, self.predict_config)
                prediction.data.to_csv(self.test_output)
                artifacts["test_output"] = self.test_output
            eval_results = model.simple_evaluate(test_dataset, self.predict_config)
            log_to_mlflow(self.dict(), history.history|eval_results, artifacts, 
                        experiment_name=self.experiment_name,
                        run_name=self.run_name, tags=self.tags)
        else:
            self.dataset_spec.data_source = self.test_path
            test_dataset = TimeSeriesDataset(_dataset_spec=self.dataset_spec)
            model = get_forecasting_model(self.training_config.model_class)(self.dataset_spec)
            model.set_params(self.training_config.model_parameters)
            model.load_model(self.training_config.model_save_folder, self.training_config, test_dataset)
            model.visualize(self.training_config.model_save_folder)
            artifacts = {}
            if self.test_output:
                prediction = model.simple_predict(test_dataset, self.predict_config)
                prediction.data.to_csv(self.test_output)
                artifacts["test_output"] = self.test_output
            
            eval_results = model.simple_evaluate(test_dataset, self.predict_config)
            log_to_mlflow(self.dict(), eval_results, artifacts)
