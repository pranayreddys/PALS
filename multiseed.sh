#!/bin/bash

for seed in {1..5}
do
echo "$seed"
python3 driver.py --config_path configs/config_scaled.json --model_save_folder logs/seed"$seed" --seed "$seed" --experiment_name seed
done