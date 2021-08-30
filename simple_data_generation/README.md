# Data Generation

There are two main files here for data generation: [gen.py](gen.py) and [generate_based_on_user_category.py](generate_based_on_user_category.py). The files here are created in the context of Study-1 data of PALS, i.e. there are several "users" created, and each user goes through a cycle of nudges.

This directory is independent of the directories outside this folder.

1. [gen.py](gen.py) is a general data generation file, which generates a) People specific baseline UAT/BP from population level distributions, b) Updates UAT based on changes in Nudge and BP based on changes in UAT, c) Has support for parameters such as inertia and psychological response (although currently they are being fixed to a constant). An example configuration to [gen.py](gen.py) is [config.json](config.json). In this file, it is assumed that effect profile is shared across the population. The file can be run with `python3 gen.py --config_path $path --output_folder $folder_path`. The output consists of two csvs, data.csv and effect_profile.csv. data.csv contains the generated data, and effect_profile.csv contains the delayed effect curve and other hidden parameters utilized for data generation.

2. [generate_based_on_user_category.py](generate_based_on_user_category.py) is a file that is used to generate two personas based on a user feature "category", with the assumption that there are only two nudges. Each persona has different psychological response (`inertia` within the code), and thus group of people end up with a different optimal nudge. This file was created with the sole intention of rapid experimentation and is not intended to be a general data creation module, for general persona based effect profile generation please update the file [gen.py](gen.py). The output is similar to [gen.py](gen.py)

## Helper Script for Converting to Nudge Evaluation Format
[split_for_reward_predictor.ipynb](split_for_reward_predictor.ipynb) is a python notebook that splits the time series dataset into Train and Test, and further converts the Test dataset into the Nudge Evaluation format. 