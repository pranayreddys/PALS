import numbers
import os
import mlflow
from entities.experiment_config import ExperimentParams, ExperimentMetrics, ExperimentOutputArtifacts
from utils.data_manipulation_utils import flatten

#TODO: need to set this up
# TRACKING_URL = ""


def log_to_mlflow(params, metrics, artifact_dict, tags=None, experiment_name="Default", run_name=None):
    """Logs metrics, parameters and artifacts to MLflow
    Args:
        params (dict of {str: str}): input parameters to the model
        metrics (dict of {str: numeric}): metrics output from the model
        artifact_dict (dict): file paths of artifacts
        tags (dict): dict of tags
        experiment_name (str): name of the MLflow experiment (default: "default")
        run_name (str): name of the MLflow run (default: None)
    """

    # mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(experiment_name)

    if isinstance(params, ExperimentParams):
        params = params.dict()
    if isinstance(metrics, ExperimentMetrics):
        metrics = metrics.dict()
    if isinstance(artifact_dict, ExperimentOutputArtifacts):
        artifact_dict = artifact_dict.dict()

    params = flatten(params)
    t = list(params.keys())
    for key in t:
        if ":" in key:
            params[key.replace(":", "_")] = params.pop(key)
    
    metrics = flatten(metrics)
    artifact_dict = flatten(artifact_dict)

    with mlflow.start_run(run_name=run_name):
        for k,v in params.items():
            mlflow.log_params({k:v})
        for metric, value in metrics.items():
            if isinstance(value, numbers.Number):
                mlflow.log_metric(key=metric, value=value)
            elif (isinstance(value, list) and len(value)==1 and isinstance(value[0], numbers.Number)):
                mlflow.log_metric(key=metric, value=value[0])
            elif isinstance(value, list):
                for step, v in enumerate(value):
                    if isinstance(v, numbers.Number):
                        mlflow.log_metric(key=metric, value=v, step=step)
        for artifact, path in artifact_dict.items():
            if path is not None:
                if os.path.isfile(path):
                    mlflow.log_artifact(path)
                elif os.path.isdir(path):
                    mlflow.log_artifacts(path)
        if tags is not None:
            mlflow.set_tags(tags)
