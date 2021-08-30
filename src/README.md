# About
This folder contains all the source files utilized for compliance & BP modelling. It also contains the codes for reward predictor and nudge optimizer modules. A quick overview of the subfolders are:
1. [dataset](dataset): This folder contains the source code for interfacing with the existing datasets. Currently supports TimeSeriesDataset and NudgeOptimizationDatasets, needs more work for supporting other TabularDatasets.

2. [entities](entities):
    1. [all_enums.py](entities/all_enums.py) - Essentially contains the allowed parameters in the configuration file. Based on the enum value, the correct model/loss/optimizer is loaded and retrieved for further training.
    2. [experiment_config.py](entities/experiment_config.py) - Needs to be completed, currently has been left half implemented. The idea is to utilize this config for flexible experiment pipelines.
    3. [key_entities.py](entities/key_entities.py)  - Contains the data specs for Time Series dataset, Nudge Optimization dataset, Tabular Dataset etc (for example a time series dataset expects time column, series id column, state columns).
    4. [modeling_configs.py](entities/modeling_configs.py) - Contains code to build and retrieve custom/inbuilt loss objects, optimizers, etc. Also consists of training configs for various tasks such as time series, nudge optimization, reard predictor.

3. [models](models):
    1. [models.py](entities/models.py) - Has multiple VAR and Delayed Effect models for different settings - UAT & BP both predicted within the same model.
    2. [base_models.py](entities/base_models.py) - Base models, from which other Time Series models, nudge optimization models and reward predictors models inherit from
    3. [reward_predictor_models.py](entities/reward_predictor_models.py) - Code for reward predictor models that are utilized in counterfactual evaluation present here.
    4. [simple_transformations.py](entities/simple_transformations.py) - Preprocessing such as one-hot encoding, MinMax Scaling, and other sklearn transforms are provided here.
    5. [estimators.py] - Consists of estimators such as IPS, Doubly Robust Estimator, Self Normalized Estimator for counterfactual evaluations.
    6. [nudge_optimizers.py] - Contains few sample policies for nudge optimization.

4. [publishers](publishers): The directory contains helper tools for logging, currently supports mlflow logging. If required, tensorboard could be added in the future.

5. [runners](runners): Common configuration of experiments implemented here. Current support is for two runners, a runner for Time Series training and another runner for Nudge Evaluation.

6. [utils](utils): Utilities that are quite commonly used across projects. Also contains Sharut's EDA code, it is not currently used within runners but can be hooked up.