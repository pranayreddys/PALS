import numpy as np
import pandas as pd
from entities.key_entities import LossFunctionSpec
import keras.metrics as km


def evaluate(ground_truth: pd.DataFrame, predictions: pd.DataFrame, loss_functions: List[LossFunctionSpec],
	weights: pd.DataFrame = None):
	#TODO: adding this here since we are interested in evaluating multiple loss functions and need some additional control
	#sanity check on ground truth and predictions size and weights if available
	#loop over loss functions 
	#check if computation is column wise or cell wise; 
  	#compute accordingly and aggregate
	return loss_values
