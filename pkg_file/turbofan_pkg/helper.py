""" Libraries to be used along turbofan dataset project """
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

import re, time, os
import umap, hdbscan, operator

from os.path import basename, join
from glob import glob
from pandas_summary import DataFrameSummary
from mpl_toolkits.mplot3d import Axes3D
#from MulticoreTSNE import MulticoreTSNE as TSNE
#from skopt import gp_minimize
#from xgboost import XGBRegressor, XGBClassifier, plot_importance
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from importlib import reload

from sklearn.preprocessing import scale, StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer, matthews_corrcoef, mean_absolute_error

#from keras.optimizers import Adam
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM, Dropout, TimeDistributed, BatchNormalization, Input, Bidirectional

from tqdm import *

""" Functions to be used along turbofan dataset project """

def get_RUL(dataframe, Lifetime):
    """
    dataframe - Dataframe with data 

    Lifetime - Dataframe with maximum life time (number of cycles) for each engine identified by dataset_id and unit_id

    return - A new column with the RUL for each engine, by subtracting the lifetime of each engine by the current operational cycle  
    """

    return  Lifetime.loc[(dataframe['dataset_id'], dataframe['unit_id'])] - dataframe['cycle']

def multiclass_problem(dataframe, upper_bound=20, lower_bound=5):
    """
    Function to give a class determined by the RUL of the engine. this function will give a class of 1, 2 or 0.
    2 - Bigger then upper bound
    1 - Between upper and lower bound
    0 - Less then lower bound

    dataframe - Dataframe with RUL column
    upper_bound - int with upper bound
    lower_bound  - int with lower bound
    return - Class column determined by the RUL column. Specified by the upper and lower bound
    """

    if dataframe['RUL'] > upper_bound: return int(2)
    if dataframe['RUL'] <= upper_bound and dataframe['RUL'] >lower_bound : return int(1)
    if dataframe['RUL'] <= lower_bound: return int(0)

def plot_figure(data, colors=None, alpha=1, figure_size=(10,10), s=7, cmap='jet', vmax_normalization=95, mark=',', edge=0):    
    """
    Function created to plot two dimensional data where a color is given to each point in order to identify it. 
    Mostly used in PCA, TSNE and UMAP.

    data - array-like data
    """
    #plt.figure(figsize=figure_size)
    fig = plt.scatter(data[:,0], data[:,1],
                c=colors, # set colors of markers
                cmap=cmap, # set color map of markers
                alpha=alpha, # set alpha of markers
                marker=mark, # use smallest available marker (square)
                s=s, # set marker size. single pixel is 0.5 on retina, 1.0 otherwise
                lw=edge, # don't use edges
                edgecolor='', # don't use edges
                vmax=np.percentile(colors, vmax_normalization)) # normalize luminance data
    
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.colorbar()

def HDBScan_clustering(data, test, columns, minimum_cluster_size=3000):
    """
    Clustering function using HDBscan. The function will create a new columns in the train and test set dataframes.

    data - train dataframe to be to performed clustering and then create a new column with the cluster identification
    columns - columns to perform clustering
    
    test - test dataframe where it will be created a new column with the cluster identification based on the training set clustering
    """

    clusterer = hdbscan.HDBSCAN(min_cluster_size=minimum_cluster_size, prediction_data=True).fit(data[columns])
    
    train_labels, strengths = hdbscan.approximate_predict(clusterer, data[columns])
    test_labels, strengths = hdbscan.approximate_predict(clusterer, test[columns])
    
    print('Number of clusters in training set using HDBScan: {}'.format(len(np.unique(train_labels))))
    print('Number of clusters in test set using HDBScan: {}'.format(len(np.unique(train_labels))))

    data['HDBScan'] = train_labels
    test['HDBScan'] = test_labels

def error_function(df, y_predicted, y_true):
    return int(df[y_predicted] - df[y_true])


def score_function(df, label, alpha1=13, alpha2=10):
    if df[label] < 0:
        return (np.exp(-(df[label] / alpha1)) - 1)

    elif df[label] >= 0:
        return (np.exp((df[label] / alpha2)) - 1)


def timeseries_difference(timeseries, diff_lag=1):
    """
    Differencing of a timeseries with a predifined lag

    timeseries - Series type to be differentiated

    return - Differenced timeseries with fillna with 0 value
    """
 
    diff = timeseries.diff(diff_lag)
    return diff.fillna(0)

def logarithmic_scaling(timeseries, flag=True):
    """
    Apply logarithmic scaling

    timeseries - The series to apply the scaling. Series or numpy array
    flag - if True then apply log(1+x)

    return - The scaled series
    """
    if flag:
        timeseries_log = timeseries.apply(lambda x: np.log(x+1))
    else:
        timeseries_log = timeseries.apply(lambda x: np.log(x))

    return timeseries_log.fillna(0)

def test_stationarity(timeseries, windown_size=7, plot_graph=False, figure_size=(40,20)):
    """
    Function to test a analyze time-series behaviour. 
    Plot rolling mean, ACF and PACF graphs to analyze AR and MA processes.
    """

    if plot_graph == True:
        plt.figure(figsize=figure_size)

        plt.subplot(221)
        plt.title('Time-Series')
        timeseries.plot();

        plt.subplot(222)
        plt.title('Time-Series with rolling mean of {}'.format(windown_size))
        timeseries.rolling(window=windown_size, center=False).mean().plot();

        plt.subplot(223)
        plt.title('Auto correlation function plot')
        pd.Series(acf(timeseries, nlags=len(timeseries))).plot();

        plt.subplot(224)
        plt.title('Partial auto correlation function plot')
        pd.Series(pacf(timeseries, nlags=len(timeseries))).plot();

    
    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)

def print_results(model, train_X, train_Y, test_X, test_Y, size, df_test, df_train, keras=True):
    """
    Print results of model. MSE and competetion score
    """
    
    if keras:
        predictions = model.predict(test_X, batch_size = size)
        predictions_train = model.predict(train_X, batch_size = size)
    else:
        predictions = model.predict(test_X)
        predictions_train = model.predict(train_X)
    
    df_predictions = df_test.join(pd.DataFrame(predictions.round()))
    df_predictions_train = df_train.join(pd.DataFrame(predictions_train.round()))

    df_predictions['error'] = df_predictions.apply(lambda df: error_function(df), axis=1)
    df_predictions_train['error'] = df_predictions_train.apply(lambda df: error_function(df), axis=1)

    df_predictions['score'] = df_predictions.apply(lambda df: score_function(df), axis=1)
    df_predictions_train['score'] = df_predictions_train.apply(lambda df: score_function(df), axis=1)

    MSE = mean_squared_error(df_predictions['RUL'], df_predictions[0])
    MSE_train = mean_squared_error(df_predictions_train['RUL'], df_predictions_train[0])

    print('Mean Squared Error on all test set: {0}'.format(MSE))
    print('Mean Squared Error on all training set: {0}'.format(MSE_train))

    df_target = df_predictions.sort_values('RUL', ascending=True).drop_duplicates(['dataset_id','unit_id']).sort_index().copy()
    df_target_train =  df_predictions_train.sort_values('RUL', ascending=True).drop_duplicates(['dataset_id', 'unit_id']).sort_index().copy()

    score_test = df_target['score'].sum()
    score_train = df_target_train['score'].sum()

    print('Score on test set: {0}'.format(score_test))
    print('Score on training set: {0}'.format(score_train))
    
def RUL_by_parts(df, RUL=150):
    """
    Calculate piecewise RUL
    """
    if df['RUL'] > RUL: return RUL
    if df['RUL'] <= RUL: return df['RUL']
    
def eval_score(y_true, y_pred):
    """
    Function to use on XGBoost score function - Competition function
    """
    error = y_true-y_pred
    
    erro = []
    for errors in error:
        if errors < 0: x = (np.exp((-errors/13))-1)
        if errors >= 0: x = (np.exp((errors/20))-1)
        erro.append(x)
    
    return -np.sum(erro)

def cat_to_continuous_Binarizer(df_train, df_test):
    
    setting_operational = ["setting_op {}".format(s) for s in range(1,8)]
    dataset_id_columns = ["dataset_id {}".format(s) for s in range(1,5)]

    preprocess_HDBscan = LabelBinarizer()
    preprocess_ID = LabelBinarizer()

    preprocess_HDBscan.fit(df_train['HDBScan'])
    preprocess_ID.fit(df_train['dataset_id'])

    dataframe_HDBscan = pd.DataFrame(preprocess_HDBscan.transform(df_train['HDBScan']), columns=setting_operational)
    dataframe_dataset_id = pd.DataFrame(preprocess_ID.transform(df_train['dataset_id']), columns=dataset_id_columns)

    dataframe_HDBscan_test = pd.DataFrame(preprocess_HDBscan.transform(df_test['HDBScan']), columns=setting_operational)
    dataframe_dataset_id_test = pd.DataFrame(preprocess_ID.transform(df_test['dataset_id']), columns=dataset_id_columns)

    df_train = df_train.join(dataframe_HDBscan) 
    df_train = df_train.join(dataframe_dataset_id)
    df_test = df_test.join(dataframe_HDBscan_test)
    df_test = df_test.join(dataframe_dataset_id_test)
    
    return df_train, df_test

def cat_to_continuous_Encoder(df_train, df_test):
    
    setting_operational = ['setting_op_one_hot']
    dataset_id_columns = ['dataset_id_one_hot']

    preprocess_HDBscan = LabelEncoder()
    preprocess_ID = LabelEncoder()

    preprocess_HDBscan.fit(df_train['HDBScan'])
    preprocess_ID.fit(df_train['dataset_id'])

    dataframe_HDBscan = pd.DataFrame(preprocess_HDBscan.transform(df_train['HDBScan']), columns=setting_operational)
    dataframe_dataset_id = pd.DataFrame(preprocess_ID.transform(df_train['dataset_id']), columns=dataset_id_columns)

    dataframe_HDBscan_test = pd.DataFrame(preprocess_HDBscan.transform(df_test['HDBScan']), columns=setting_operational)
    dataframe_dataset_id_test = pd.DataFrame(preprocess_ID.transform(df_test['dataset_id']), columns=dataset_id_columns)

    df_train = df_train.join(dataframe_HDBscan) 
    df_train = df_train.join(dataframe_dataset_id)
    df_test = df_test.join(dataframe_HDBscan_test)
    df_test = df_test.join(dataframe_dataset_id_test)

    return df_train, df_test
    
def print_results(model, keras=True):
    """
    Print results of model. MSE and competetion score
    """

    if keras:
        predictions = model.predict(test_X, batch_size = batch_size)
        predictions_train = model.predict(train_X, batch_size = batch_size)
    else:
        predictions = model.predict(test_x)
        predictions_train = model.predict(train_x)
    
    df_predictions = df_test.join(pd.DataFrame(predictions.round()))
    df_predictions_train = df_train.join(pd.DataFrame(predictions_train.round()))
    
    df_predictions.rename(columns={0: 'Prediction'}, inplace=True)
    df_predictions_train.rename(columns={0: 'Prediction'}, inplace=True)

    df_predictions['error'] = df_predictions.apply(lambda df: error_function(df), axis=1)
    df_predictions_train['error'] = df_predictions_train.apply(lambda df: error_function(df), axis=1)

    df_predictions['score'] = df_predictions.apply(lambda df: score_function(df), axis=1)
    df_predictions_train['score'] = df_predictions_train.apply(lambda df: score_function(df), axis=1)

    MSE_all = mean_squared_error(df_predictions['RUL'], df_predictions['Prediction'])
    MSE_train_all = mean_squared_error(df_predictions_train['RUL'], df_predictions_train['Prediction'])
    
    print('{:.2f}'.format(MSE_all))
    print('{:.2f}'.format(MSE_train_all))
    
    MAE_all = mean_absolute_error(df_predictions['RUL'], df_predictions['Prediction'])
    MAE_train_all = mean_absolute_error(df_predictions_train['RUL'], df_predictions_train['Prediction'])
    
    print('{:.2f}'.format(MAE_all))
    print('{:.2f}'.format(MAE_train_all))

    ##################
    
    temp_df = pd.DataFrame(df_predictions.groupby(['dataset_id', 'unit_id'])['cycle'].max()).reset_index()
    df_target = pd.merge(temp_df, df_predictions, how='left', on=('dataset_id', 'unit_id', 'cycle'))

    temp_df = pd.DataFrame(df_predictions_train.groupby(['dataset_id', 'unit_id'])['cycle'].max()).reset_index()
    df_target_train = pd.merge(temp_df, df_predictions_train, how='left', on=('dataset_id', 'unit_id', 'cycle'))
    
    MSE = mean_squared_error(df_target['RUL'], df_target['Prediction'])
    MSE_train = mean_squared_error(df_target_train['RUL'], df_target_train['Prediction'])
    
    print('{:.2f}'.format(MSE))
    print('{:.2f}'.format(MSE_train))
    
    MAE = mean_absolute_error(df_target['RUL'], df_target['Prediction'])
    MAE_train = mean_absolute_error(df_target_train['RUL'], df_target_train['Prediction'])
    
    print('{:.2f}'.format(MAE))
    print('{:.2f}'.format(MAE_train))
    
    score_test = df_target['score'].sum()
    score_train = df_target_train['score'].sum()

    print('{:.2f}'.format(score_test))
    print('{:.2f}'.format(score_train))
    
    return df_predictions, df_predictions_train, df_target, df_target_train