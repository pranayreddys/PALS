import enum

""" This file contains the types allowed in the configuration files.
Whenever a new type needs to be implemented, it also needs to be added here.

"""
@enum.unique
class OptimizerType(str, enum.Enum):
	"""
	This enum is used in the Optimizer definition.
	By default it calls the tensorflow implementation directly - if there further 
	tensorflow optimizers that need to be added, just writing the optimizer type here is enough.
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
	List of supported inbuilt loss metrics. Any other keras losses can just be added directly here if
	no additional parameters are required.
	"""
	MeanSquaredError = 'MeanSquaredError'
	MeanAbsoluteError = 'MeanAbsoluteError'
	MeanAbsolutePercentageError = 'MeanAbsolutePercentageError'
	MeanSquaredLogarithmicError = 'MeanSquaredLogarithmicError'
	

@enum.unique
class TimeUnit(str, enum.Enum):
	"""TODO: Not utilized yet. 
	"""
	Second = 'second'
	Minute = 'minute'
	Hour = 'hour'
	Day = 'day'

@enum.unique
class ModelClass(str, enum.Enum):
	"""
	List of valid class names for modelling. Any additional models that are implemented in models.models.py
	need to be added here.
	"""
	SimpleVAR = "SimpleVAR"
	UatVAR = "UatVAR"
	UatBpVAR = "UatBpVAR"
	DelayedEffectModel = "DelayedEffectModel"
	BpDelayedEffectModel = 'BpDelayedEffectModel'
	BpDelayedEffectModelUserFeature = 'BpDelayedEffectModelUserFeature'


@enum.unique
class NudgeOptimizerModelClass(str, enum.Enum):
	"""
	List of valid class names for nudge optimization. Additional implemented optimizers need to be added here.
	"""
	NudgeOptimizerToy = "NudgeOptimizerToy"
	UniformActionRecommender = "UniformActionRecommender"
	NudgeOptimizerFromTimeSeries = "NudgeOptimizerFromTimeSeries"


@enum.unique
class EstimatorClass(str, enum.Enum):
	"""
		List of valid class names for estimators. Additional implemented optimizers need to be added here.
	"""
	DoublyRobustEstimator = "DoublyRobustEstimator"
	SelfNormalizedEstimator = "SelfNormalizedEstimator"
	ModelTheWorldEstimator = "ModelTheWorldEstimator"
	IPS = "IPS"

@enum.unique
class RewardPredictorClass(str, enum.Enum):
	"""
	Enum utilized in the code for different types of reward predictors. Refer models.reward_predictors.py for additional details.
	"""
	RewardPredictorToy = "RewardPredictorToy"
	RewardPredictorFromTimeSeries = "RewardPredictorFromTimeSeries"


@enum.unique
class ColumnTransform(str, enum.Enum):
	"""
	Supported column transformations, these are all implemented in simple_transformations.py
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
	Note that as a workaround, files can be presplit (i.e. explicitly constructing train.csv/val.csv/test.csv), and provided
	to the runner.


	In additon to these two modes, there is an additional presplit option, which requires different train,
	test and val files. That option is suitable when the split is predetermined, it would stop the script
	from splitting according to the two options below.
	"""
	SimpleForecast = 'SimpleForecast'
	MultiTimeSeries = 'MultiTimeSeries'
