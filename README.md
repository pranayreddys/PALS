# Health Intervention study modelling
This directory consists of the data modelling code for the PALS study. 

The primary aim of the PALS study is to provide personalized health recommendations and analyzing the effect of such interventions for people suffering from hypertension. To enable such personalized interventions, the following machine learning based tool has been created specifically targetted towards health recommendations. The tool is meant for:

1. Creating machine learning models to understand the effect of different health interventions.
2. Matching study participants to the correct health intervention based on their state.
3. Testing out and evaluating different matching strategies in an offline setting.

There are several modules and functions in place for enabling fast prototyping and testing. Currently, the tool supports four major functions:
1. Multivariate Time Series Forecasting - A big part of the study involves understanding the effect of nudges on BP across time. The time series forecasting module is built specifically for this purpose. For better interpretability of forecasting models, we take a two-step approach for time series forecasting: we perform a Nudge - User Activity prediction (compliance modelling), followed by a User Activity to BP prediction (BP progression/physiological factor modelling). The objective is to use these predictions to finally come up with reliable estimates and interventions.
2. Vertex AI based AutoML time series forecasting - In addition to tranditional time series forecasting, the tool also supports AutoML for forecasting. The models generated via AutoML are quite uninterpretable, but can still be utilized for generating strong baselines to compare our methods against. 
3. Data Generaton - A data generation framework (along with the Delayed Effect Model based generation), is provided in this code. Before deploying any models for real-world data, it is important to test for bugs. This data generation module attempts to do just that, by providing synthetic data for easy model testing across multiple different parameters and distributions. In addition, if we are able to accurately assign distributions for certain variables, it might be possible to even augment real-world data with synthetic data for better forecasting accuracies.
4. Counterfactual Evaluation - Before deploying the model, it is vital to estimate the expected reward generated for our health intervention policy. However, data is quite limited and sparse, and we do not have real-world information of the counterfactuals. With counterfactual evaluation, the tool overcomes this deficiency by estimating reward factoring in the effect of counterfactuals.

## Framework utilization instructions
There are two driver scripts for running - [driver_modelling.py](driver_modelling.py) and [driver_evaluation.py](driver_evaluation.py). For both these scripts, the command to run them is ``python3 driver.py --config_path path_to_config``. For sample configs, please take a look at the configs folder. The first script is for modelling purposes, while the second script is for counterfactual evaluation.

Similar scripts exist in the simple data generation folder for data generation ([gen.py](simple_data_generation/gen.py)) along with the config ([config.json](simple_data_generation/config.json))

In addition, there are similar python notebooks for interfacting with AutoML ([Training_automl.ipynb](notebooks/Training_automl.ipynb)) along with the config ([training_automl_config.json](notebooks/training_automl_config.json)).