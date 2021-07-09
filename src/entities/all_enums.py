import enum

@enum.unique
class OptimizerType(str, enum.Enum):
	Adam = 'adam'
	SGD = 'sgd'	
	RMSprop = 'rmsprop'
	Adadelta = 'adadelta'
	Adagrad = 'adagrad'
	Adamax = 'adamax'
	Nadam = 'nadam'
	Ftrl = 'ftrl'

@enum.unique
class Distribution(str, enum.Enum):
	#TODO
	pass

@enum.unique
class LossMetric(str, enum.Enum):
	MeanSquaredError = 'mse'
	MeanAbsoluteError = 'mae'
	MeanAbsolutePercentageError = 'mape'
	MeanSquaredLogarithmicError = 'msle'
	

@enum.unique
class TimeUnit(str, enum.Enum):
	pass
	#TODO

@enum.unique
class ModelCallMode(str, enum.Enum):
	pass
	#TODO - this will different modes of model call
	#call implementation might be different during prediction and fitting

