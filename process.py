import re, os
import numpy as np
import pandas as pd
import pickle as pkl
import argparse

from os.path import basename, join

import hdbscan

def get_RUL(dataframe, Lifetime):
    return  Lifetime.loc[(dataframe['dataset_id'], dataframe['unit_id'])] - dataframe['cycle']


def RUL_by_parts(df, RUL=130):
    if df['RUL'] > RUL: return RUL
    if df['RUL'] <= RUL: return df['RUL']

parser = argparse.ArgumentParser(description='Script Variables')
parser.add_argument('--data_path', default='/datadrive/Turbofan_Engine/', type=str,
                        help='Folder with txt files')
args = parser.parse_args()

datasets = []
path =  args.data_path # path to .txt files
text_files = [f for f in os.listdir(path) if f.endswith('.txt') and not f.startswith('r')]
dataframe = [os.path.splitext(f)[0] for f in text_files]
sensor_columns = ["sensor {}".format(s) for s in range(1, 22)]
info_columns = ['dataset_id', 'unit_id', 'cycle', 'setting 1', 'setting 2', 'setting 3']
label_columns = ['dataset_id', 'unit_id', 'rul']
settings = ['setting 1', 'setting 2', 'setting 3']

test_data = []
train_data = []
RUL_data = []

for file in text_files:
    print(file)

    if re.match('RUL*', file):
        subset_df = pd.read_csv(path + file, delimiter=r"\s+", header=None)
        unit_id = range(1, subset_df.shape[0] + 1)
        subset_df.insert(0, 'unit_id', unit_id)
        dataset_id = basename(file).split("_")[1][:5]
        subset_df.insert(0, 'dataset_id', dataset_id)
        RUL_data.append(subset_df)

    if re.match('test*', file):
        subset_df = pd.read_csv(path + file, delimiter=r"\s+", header=None, usecols=range(26))
        dataset_id = basename(file).split("_")[1][:5]
        subset_df.insert(0, 'dataset_id', dataset_id)
        test_data.append(subset_df)

    if re.match('train*', file):
        subset_df = pd.read_csv(path + file, delimiter=r"\s+", header=None, usecols=range(26))
        dataset_id = basename(file).split("_")[1][:5]
        subset_df.insert(0, 'dataset_id', dataset_id)
        train_data.append(subset_df)

df_train = pd.concat(train_data, ignore_index=True)
df_train.columns = info_columns + sensor_columns
df_train.sort_values(by=['dataset_id', 'unit_id', 'cycle'], inplace=True)

df_test = pd.concat(test_data, ignore_index=True)
df_test.columns = info_columns + sensor_columns
df_test.sort_values(by=['dataset_id', 'unit_id', 'cycle'], inplace=True)

df_RUL = pd.concat(RUL_data, ignore_index=True)
df_RUL.columns = label_columns
df_RUL.sort_values(by=['dataset_id', 'unit_id'], inplace=True)

RUL_train = df_train.groupby(['dataset_id', 'unit_id'])['cycle'].max()
RUL_test = df_test.groupby(['dataset_id', 'unit_id'])['cycle'].max() + df_RUL.groupby(['dataset_id', 'unit_id'])[
    'rul'].max()

df_train['RUL'] = df_train.apply(lambda r: get_RUL(r, RUL_train), axis=1)
df_test['RUL'] = df_test.apply(lambda r: get_RUL(r, RUL_test), axis=1)

df_train['RUL'] = df_train.apply(lambda r: RUL_by_parts(r, 130), axis=1)
df_test['RUL'] = df_test.apply(lambda r: RUL_by_parts(r, 130), axis=1)

clusterer = hdbscan.HDBSCAN(min_cluster_size=3000, prediction_data=True).fit(df_train[['setting 1', 'setting 2', 'setting 3']])

train_labels, strengths = hdbscan.approximate_predict(clusterer, df_train[['setting 1', 'setting 2', 'setting 3']])
test_labels, strengths = hdbscan.approximate_predict(clusterer, df_test[['setting 1', 'setting 2', 'setting 3']])

df_train['HDBScan'] = train_labels
df_test['HDBScan'] = test_labels

df_train.set_index(['dataset_id', 'unit_id'], inplace=True)
df_test.set_index(['dataset_id', 'unit_id'], inplace=True)

pd.to_pickle(df_train, args.data_path + '/df_train_cluster_piecewise.pkl')
pd.to_pickle(df_test, args.data_path + '/df_test_cluster_piecewise.pkl')
