import enum

@enum.unique
class OptimizerType(str, enum.Enum):
	Adam = 'Adam'
	SGD = 'SGD'	
	RMSprop = 'RMSprop'
	Adadelta = 'Adadelta'
	Adagrad = 'Adagrad'
	Adamax = 'Adamax'
	Nadam = 'Nadam'
	Ftrl = 'Ftrl'

@enum.unique
class DistributionType(str, enum.Enum):
	Normal = 'Normal'
	Bernoulli = 'Bernoulli'
	Beta = 'Beta'
	Gamma = 'Gamma'
	Binomial = 'Binomial'
	Multinomial = 'Multinomial'

@enum.unique
class LossMetric(str, enum.Enum):
	MeanSquaredError = 'MeanSquaredError'
	MeanAbsoluteError = 'MeanAbsoluteError'
	MeanAbsolutePercentageError = 'MeanAbsolutePercentageError'
	MeanSquaredLogarithmicError = 'MeanSquaredLogarithmicError'
	

@enum.unique
class TimeUnit(str, enum.Enum):
	Second = 'second'
	Minute = 'minute'
	Hour = 'hour'
	Day = 'day'
	#TODO

@enum.unique
class ModelClass(str, enum.Enum):
	SimpleVAR = "SimpleVAR"
	pass

@enum.unique
class ModelCallMode(str, enum.Enum):
	pass
	#TODO - this will different modes of model call
	#call implementation might be different during prediction and fitting

@enum.unique
class ColumnTransform(str, enum.Enum):
	OneHotEncoder = 'OneHotEncoder'
	MinMaxScaler = 'MinMaxScaler'
	StandardScaler = 'StandardScaler'
	DateTime = 'DateTime'
	Identity = 'Identity'

@enum.unique
class ExperimentMode(str, enum.Enum):
	SimpleForecast = 'SimpleForecast'
	MultiTimeSeries = 'MultiTimeSeries'
