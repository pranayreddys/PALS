# Notebooks
## AutoML Notebooks
There are two notebooks provided here. These notebooks are designed to easily create and test AutoML Time Series models. Although there is no control over model selection, AutoML can serve as a very strong baseline on achievable results for time series data. 

The notebooks can directly be run from Google Cloud Platform without additional setups other than environment initialization.

1. [Training_automl.ipynb](Training_automl.ipynb) - The notebook is used for creating and training an AutoML model. The network architecture, ensembling etc are automatically determined by AutoML itself. This notebook reads the configuration from [training_automl_config.json](training_automl_config.json). Detailed explanations of all the parameters are provided within as comments within the notebook. 

2. [Predict_automl.ipynb](Predict_automl.ipynb)  - This notebook reads the configuration from [predict_automl_config.json](predict_automl_config.json). Detailed explanations of all the parameters are provided within as comments within the notebook.

## Prediction Visualization
[modelling_test.ipynb](modelling_test.ipynb) is a very rudimentary notebook meant for visualizing the prediction of time series models. The notebook can be improved based on the required functionality in the future. 