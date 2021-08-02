#!/bin/bash

for file in configs/data_variation/*
do
python3 driver.py --config_path "$file"
done