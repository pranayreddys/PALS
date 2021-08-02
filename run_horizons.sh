#!/bin/bash

for file in configs/forecasting_horizons/*
do
python3 driver.py --config_path "$file"
done