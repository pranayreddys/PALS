import enum

@enum.unique
class Distribution(str, enum.Enum):
	#TODO

@enum.unique
class LossMetric(str, enum.Enum):
	#TODO

@enum.unique
class TimeUnit(str, enum.Enum):
	#TODO

@enum.unique
class ModelCallMode(str, enum.Enum):
	#TODO - this will different modes of model call
	#call implementation might be different during prediction and fitting

