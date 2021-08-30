import argparse
from pydantic import parse_file_as
import sys
sys.path.append("src")
from runners.policy_evaluator import Runner
from utils.read_write_utils import is_file
from numpy.random import seed
import tensorflow as tf
import random
import os

def set_seed(seed_val: int):
    seed(seed_val)
    random.seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    tf.random.set_seed(seed_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=is_file)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    r = parse_file_as(Runner, args.config_path)
    r()
    