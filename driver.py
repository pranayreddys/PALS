import argparse
from pydantic import parse_file_as
import sys
sys.path.append("src")
from runners.model_runner import Runner
from utils.read_write_utils import is_file
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(3)
import os
os.environ['PYTHONHASHSEED'] = '0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=is_file)
    parser.add_argument('--experiment_name', type=str)
    args = parser.parse_args()
    r = parse_file_as(Runner, args.config_path)
    if args.experiment_name:
        r.experiment_name = args.experiment_name
    r.run()
    