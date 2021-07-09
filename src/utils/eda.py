"""Exploratory Data Analysis on Time Series Data.

Author: Sharut Gupta <sharut@google.com>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity
from statsmodels.tsa import api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
sns.set()
warnings.filterwarnings('ignore')


# Baseline system settings
SEED = 42
np.random.seed(SEED)
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (20, 7)
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'

# Model controllers
plot_bool = True
signif_level = 0.05


def statistics(input_df: pd.DataFrame) -> pd.DataFrame:
  """Central tendency measures descibing the input.


  Utility function to calculate some basic statistical
  details like percentile, mean, std etc of a data
  frame or a series of numeric values.

  Args:
    input_df : pandas dataframe of shape (n_samples, n_features)
              or pandas series
              First input samples

  Returns:
    measures : pandas dataframe  of shape (, n_features)
             Statistical summary of data frame or series
  """
  return input_df.describe()


def visualize(df: pd.DataFrame) -> None:
  """Visualize the input time series.

  Args:
    df: Pandas Dataframe object
        Furst Input Samples

  Returns:
    Plots of all series in the input dataframe as subplots
  """

  if not plot_bool:
    return

  # Plot all columns of the given dataframe
  plt.figure(figsize=(18, 10))
  values = df.values
  i = 1
  for group in range(df.shape[1]):
    plt.subplot(df.shape[1], 1, i)
    plt.plot(df.index, values[:, group])
    plt.title(df.columns[group], y=0.5, loc='right')
    i += 1
  plt.tight_layout()
  plt.show()
  plt.close('all')


def intracorrelation(input_df: pd.DataFrame,
                     plot: bool = True) -> dict[str, pd.DataFrame]:
  """Calculates Intra-Feature correlation matrix.


  Feature correlation function to calculate pearson, kendall
  and spearman correlation between all pairwise features in the
  input dataframe.

  Args:
    input_df : pandas dataframe of shape (n_samples, n_features)

    plot : bool
         indicates toggle between plot visualization of
         intracorrelation matrix

  Returns:
    corr_list : dictionary of length 3
              dictionary containing Pearson, Kendal and Spearman
              Correlation coefficient matrix on the input dataframe
              features

  """
  corr_list = {}
  if not plot:
    return
  for corr_type in ['pearson', 'kendall', 'spearman']:
    corr = input_df.corr(method=corr_type)
    corr_list[corr_type] = corr
    if not plot:
      continue

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

    for (i, j), z in np.ndenumerate(corr):
      ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    fig.colorbar(cax)
    ticks = np.arange(0, len(input_df.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(input_df.columns)
    ax.set_yticklabels(input_df.columns)
    plt.tight_layout()
    plt.show()

  return corr_list


def grangers_causation_matrix(input_df: pd.DataFrame,
                              test_type: str = 'ssr_chi2test', maxlag: int = 30,
                              verbose: bool = True,
                              plot: bool = True) -> pd.DataFrame:
  """Granger’s causality tests to study cause effect relationship.


  Granger’s causality tests to investigate causality between two
  variables in a time series. This is done by testing the null
  hypothesis that the coefficients of past values of a series x
  (cause) in the regression of a series y (effect) are 0

  This utility function checks Granger Causality of all possible
  combinations of the time series.

  Args:

    input_df : pandas dataframe of shape (n_samples, n_features)
    test_type : string out of ['ssr_ftest', 'ssr_chi2test', 'lrtest',
      'params_ftest'] string representing the type of test results we want to
      consider.
    maxlag : int maximum lag/history that needs to be considered to optimize
      casuality between two time series objects
    verbose : bool toggles between printing the optimization output of granger's
      casuality to stdout
    plot: bool indicates toggle between plot visualization of causation matrix

  Returns:
    df : pandas dataframe of shape (n_features, n_features)
       Output table with rows as the  response variable, columns as
       predictors. The values in the table denote the p-values. p-values
       lesser than the significance level (0.05), implies he null hypothesis
       that the coefficients  of the corresponding past values is  zero,
       that is, the X does not cause Y can be rejected.

  """
  variables = input_df.columns
  df = pd.DataFrame(np.zeros((len(variables), len(variables))),
                    columns=variables, index=variables)
  for c in df.columns:
    for r in df.index:
      test_result = grangercausalitytests(input_df[[r, c]],
                                          maxlag=maxlag, verbose=verbose)
      p_values = [round(test_result[i+1][0][test_type][1], 4) for i in range(maxlag)]
      min_p_value = np.min(p_values)
      df.loc[r, c] = min_p_value
  df.columns = [var + '_x' for var in variables]
  df.index = [var + '_y' for var in variables]

  if not plot:
    return df

  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111)
  cax = ax.matshow(df, cmap='coolwarm', vmin=-1, vmax=1)
  for (i, j), z in np.ndenumerate(df):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
  fig.colorbar(cax)
  ticks = np.arange(0, len(df.columns), 1)
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_xticklabels(df.columns)
  ax.set_yticklabels(df.index)
  plt.show()


def decomposition(input_df: pd.DataFrame,
                  decomposition_type: str = 'multiplicative',
                  period: int = 60, plot: bool = True):
  """Decomposes the given time series into 4 components.


  Time series decomposition that splits a time series into several
  components, each representing an underlying pattern category,
  trend, seasonality, and noise/residuals. The results are obtained
  by first estimating the trend by applying a convolution filter to
  the data. The trend is then removed from the series and the average
  of this de-trended series for each period is the returned seasonal component.
  Level - The average value in the series.
  Trend - The increasing or decreasing value in the series.
  Seasonality - The repeating short-term cycle in the series.
  Noise - The random variation in the series.

  Args:
    input_df : pandas series of shape (n_samples,1)
      First input data

    decomposition_type : string from ['multiplicative', 'additive']
                       represents the type of decomposition.
                       Additive => X(t) = Level + Trend + Seasonality + Noise
                       Multiplicative => X(t) = Level * Trend *
                       Seasonality * Noise

    period : int
           Period of the series X. Must be used if x is not a pandas object
           or if the index of x does not have a frequency

    plot: bool
        indicates toggle between plot visualization of causation matrix


  Returns:
    decomposed : DecomposeResult object
               object with seasonal, trend, and resid attributes.

  """

  decomposed = seasonal_decompose(input_df, model=decomposition_type,
                                  period=period)
  if not plot:
    return decomposed

  plt.figure(figsize=(20, 7), dpi=300)
  decomposed.plot()
  plt.show()
  plt.close('all')
  return decomposed


def adfuller_test(input_df: pd.DataFrame, signif: float = 0.05) -> None:
  """The Augmented Dickey-Fuller test or the unit root test.


  This determine how strongly a time series is defined by a trend.
  This is done using an autoregressive model and optimizes an
  information criterion across multiple different lag values.
  The null hypothesis is that time series is not stationary

  Args:
    input_df : pandas series of shape (n_samples,1)
      First input data

    signif : float
           significance level for the test

  Returns:
    r : prints a report of  Augmented Dickey-Fuller test
      reporting stationarity of the series

  """

  r = adfuller(input_df.values)
  output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4),
            'n_lags': round(r[2], 4), 'n_obs': r[3]}
  p_value = output['pvalue']
  def adjust(val, length=6):
    return str(val).ljust(length)

  ### Output Summary
  print('    Augmented Dickey-Fuller Test', '\n   ', '-'*47)
  print(' Null Hypothesis: Data has unit root. Non-Stationary.')
  print(f' Significance Level    = {signif}')
  print(f' Test Statistic        = {output["test_statistic"]}')
  print(f' No. Lags Chosen       = {output["n_lags"]}')

  for key, val in r[4].items():
    print(f' Critical value {adjust(key)} = {round(val, 3)}')

  if p_value <= signif:
    print(f' => P-Value = {p_value}. Rejecting Null Hypothesis.')
    print(' => Series is Stationary.')
  else:
    print(f' => P-Value = {p_value}. Weak evidence to reject the'
          'Null Hypothesis.')
    print(' => Series is Non-Stationary.')
    print('\n')

  return


def cointegration_test(input_df: pd.DataFrame, alpha: float = 0.05) -> None:
  """The Johansen test to check cointegration among series.


  The Johansen Cointegration test to test cointegrating
  relationships between several time series data. This is a
  general and more applicable version of augmented DF test

  Args:
    input_df : pandas data frame of shape (n_samples, n_features)
      First input data

    alpha : float
           significance level for the test

  Returns:
    r : prints a report of Johansen Cointegration test
      reporting stationarity of the series
  """

  out = coint_johansen(input_df, -1, 1)
  d = {'0.90': 0, '0.95': 1, '0.99': 2}
  traces = out.lr1
  cvts = out.cvt[:, d[str(1-alpha)]]
  def adjust(val, length=6):
    return str(val).ljust(length)

  ### Output Summary
  print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
  for col, trace, cvt in zip(input_df.columns, traces, cvts):
    print(adjust(col), ':: ', adjust(round(trace, 2), 9), '>',
          adjust(cvt, 8), ' =>  ', trace > cvt)
  print('\n')
  return


def qualitative_stationarity(y: pd.Series, maxlags: int = 1000,
                             window_size: int = 7) -> None:
  """Check stationarity of the given time series qualitatively.


  Function to explore the stationarity of the input time series
  using visual plots of auto-correlation function

  Args:
    y : pandas series of shape (n_samples,1)
      First input data

    maxlags : int
            maximum lag/history for ACF plot

    window_size: int
                Sliding window size of rolling measures

  Returns:
    Plots the auto-correlation function for the specified number
    of lags along with rolling average with the given window
    size.

  """

  if not isinstance(y, pd.Series):
    y = pd.Series(y)

  with plt.style.context(style='bmh'):
    plt.figure(figsize=(20, 7))
    layout = (2, 2)
    acf_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    plt.subplot2grid(layout, (1, 0), colspan=2)

    smt.graphics.plot_acf(y, lags=maxlags, ax=acf_ax)
    rolmean = y.rolling(window=window_size).mean()
    rolstd = y.rolling(window=window_size).std()
    plt.plot(y, label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Averages, Window: ' + str(window_size))
    plt.tight_layout()
    plt.show()
    plt.close('all')


def density_estimation(x: np.ndarray, x_grid: np.ndarray,
                       kernel_type: str = 'gaussian',
                       bandwidth: float = 0.2, **kwargs) -> np.ndarray:
  """Function to estimate Kernel Density of a given series using sklearn.


  Args:
    x : array of shape (n_samples,1)
      First input data

    x_grid : array of shape (n_samples,1)
           indexing array for the samples x

    kernel_type: string
               Kernel used for estimation out of {‘gaussian’, ‘tophat’,
               ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}

    bandwidth: float
             The bandwidth of the kernel

    **kwargs : Arguments
               Arguments input to sklearn's pdf function

  Returns:
    likelihood: array of shape (n_samples,1)
              log-likelihood score of the data points in x


  """

  kde_skl = KernelDensity(kernel=kernel_type, bandwidth=bandwidth, **kwargs)
  kde_skl.fit(x[:, np.newaxis])
  log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
  return np.exp(log_pdf)


def visualise_prior_conditioning(input_df: pd.DataFrame, col: str,
                                 grouping_col: str) -> None:
  """Visualise feature distribution after conditioning on some other variable.


  Utility function to explore conditioning effects on the dataset


  Args:
    input_df : pandas dataframe of shape (n_samples, n_features)
      First input samples

    col : string
        column name of interest. This is the variable which will be
        conditioned on the grouping_col

    grouping_col : string
                 column name on which grouping will be done

  Returns:
    Plots probability density function of X[col] after prior
    conditioning based on unique values in grouping_col

  """

  plt.figure(figsize=(20, 7))
  for activity_kind, activity_df in input_df.groupby(grouping_col):
    x = activity_df[col].values

    # There will be 0 steps for intensity = 0
    if (int)(np.amax(x)) == 0: continue
    x_grid = np.array(range((int)(np.amin(x)), (int)(np.amax(x))))
    pdf = density_estimation(x, x_grid, bandwidth=0.2)
    plt.fill_between(x_grid, pdf, alpha=0.6, label=activity_kind)

  plt.title(col, fontsize=16)
  plt.legend(title='Grouping on ' + str(grouping_col))
  plt.show()


