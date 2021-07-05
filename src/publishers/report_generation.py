import pprint
import chevron
import pypandoc
import pandas as pd
from entities.experiment_config import ExperimentParams, ExperimentMetrics, \
    ExperimentOutputArtifacts
from utils.io import read_file


def create_report(params, metrics, artifact_dict, artifacts_to_render=None, template_path='template.mustache', report_path='report.md'):
    """Generates a report from a template
    Args:
        params ():
        metrics ():
        artifact_dict ():
        artifacts_to_render(dict(str, dict(str, str))):
        template_path (str): path to template file
        report_path (str): path to report generated
    """
    if isinstance(params, ExperimentParams):
        params = params.dict()
    if isinstance(metrics, ExperimentMetrics):
        metrics = metrics.dict()
    if isinstance(artifact_dict, ExperimentOutputArtifacts):
        artifact_dict = artifact_dict.dict()

    if artifacts_to_render is not None:
        for artifact, read_args in artifacts_to_render.items():
            artifact_dict[artifact] = read_file(artifact_dict[artifact], **read_args)

    params = {'_'.join(['params', k]): v for k, v in params.items()}
    metrics = {'_'.join(['metrics', k]): v for k, v in metrics.items()}
    artifact_dict = {'_'.join(['artifact_list', k]): v for k, v in artifact_dict.items()}

    template_inputs = dict()
    template_inputs.update(params)
    template_inputs.update(metrics)
    template_inputs.update(artifact_dict)

    for k, v in template_inputs.items():
        if isinstance(v, pd.DataFrame):
            template_inputs[k] = v.to_markdown()
        if isinstance(v, dict):
            template_inputs[k] = pprint.pformat(v, indent=4)

    with open(template_path, 'r') as f:
        report = chevron.render(f, template_inputs)

    f = open(report_path, 'w')
    f.write(report)
    f.close()
