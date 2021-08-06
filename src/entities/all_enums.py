import enum

@enum.unique
class OptimizerType(str, enum.Enum):
	"""
	This enum is used in the Optimizer definition
	"""
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
	"""
	List of distribution types, not used in code yet.
	"""
	Normal = 'Normal'
	Bernoulli = 'Bernoulli'
	Beta = 'Beta'
	Gamma = 'Gamma'
	Binomial = 'Binomial'
	Multinomial = 'Multinomial'

@enum.unique
class LossMetric(str, enum.Enum):
	"""
	List of supported inbuilt loss metrics.
	"""
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
	"""
	List of valid class names for modelling
	"""
	SimpleVAR = "SimpleVAR"
	UatVAR = "UatVAR"
	UatBpVAR = "UatBpVAR"
	DelayedEffectModel = "DelayedEffectModel"
	BpDelayedEffectModel = 'BpDelayedEffectModel'


@enum.unique
class ModelCallMode(str, enum.Enum):
	pass
	#TODO - this will different modes of model call
	#call implementation might be different during prediction and fitting

@enum.unique
class ColumnTransform(str, enum.Enum):
	"""
	Supported column transformations
	"""
	OneHotEncoder = 'OneHotEncoder'
	MinMaxScaler = 'MinMaxScaler'
	StandardScaler = 'StandardScaler'
	DateTime = 'DateTime'
	Identity = 'Identity'

@enum.unique
class ExperimentMode(str, enum.Enum):
	"""
	There are two possible experimental modes: Simple Forecasting for a single time series,
	and multi time series forecasting.
	Take the example of BP forecasting from UAT data. for multiple time series forecasting.
	If our aim is to predict the BP variation across days for a new user, given data
	for a set (say 300) users, then we would employ the MultiTimeSeries mode.
	In this mode, users are categorized into train, val and test splits. 
	Hence the prediction users set and the train users set are non-intersecting/disjoint.

	In the simple forecasting case, consider the time series for daily traffic variation in a 
	given city. If the idea is to forecast traffic variation given past data, then this mode needs to be employed.
	In this case, if data is provided from 0..T, time would be split into train chunks (0..T1),
	val chunk (T1..T2) and final test chunk (T2..T).
	TODO: This mode has not been implemented, refer the file :class:`~src.dataset.dataset.TimeSeriesDataset` for implementation.


	In additon to these two modes, there is an additional presplit option, which requires different train,
	test and val files. That option is suitable when the split is predetermined, it would stop the script
	from splitting according to the two options below.
	"""
	SimpleForecast = 'SimpleForecast'
	MultiTimeSeries = 'MultiTimeSeries'
