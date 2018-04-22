import os, time

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


SEED = 1337

sensor_columns = ["sensor {}".format(s) for s in range(1, 22)]

info_columns = ['dataset_id', 'unit_id', 'cycle', 'setting 1', 'setting 2', 'setting 3']

label_columns = ['dataset_id', 'unit_id', 'rul']

settings = ['setting 1', 'setting 2', 'setting 3']
type_1 = ['FD001', 'FD003']
type_2 = ['FD002', 'FD004']


class DataReader(object):
    def __init__(self,
                 raw_data_path_train,
                 raw_data_path_test,
                 **kwargs):

        assert os.path.isfile(raw_data_path_train) is True, \
            'This file do not exist. Please select an existing file'

        assert os.path.isfile(raw_data_path_test) is True, \
            'This file do not exist. Please select an existing file'

        assert raw_data_path_train.lower().endswith(('.csv', '.parquet', '.hdf5', '.pickle', '.pkl')) is True, \
            'This class can\'t handle this extension. Please specify a .csv, .parquet, .hdf5, .pickle extension'

        assert raw_data_path_test.lower().endswith(('.csv', '.parquet', '.hdf5', '.pickle', 'pkl')) is True, \
            'This class can\'t handle this extension. Please specify a .csv, .parquet, .hdf5, .pickle extension'

        self.raw_data_path_train = raw_data_path_train
        self.raw_data_path_test = raw_data_path_test
        self.loader_engine(**kwargs)
        self.train = self.loader_train()
        self.test = self.loader_test()

    def loader_engine(self, **kwargs):
        if self.raw_data_path_train.lower().endswith(('.csv')):
            self.loader_train = lambda: pd.read_csv(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_csv(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.parquet')):
            self.loader_train = lambda: pd.read_parquet(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_parquet(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.hdf5')):
            self.loader_train = lambda: pd.read_hdf(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_hdf(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.pkl', 'pickle')):
            self.loader_train = lambda: pd.read_pickle(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_pickle(self.raw_data_path_test, **kwargs)

    def calculate_unique_turbines(self):

        self.train_turbines = np.arange(len(self.train.index.to_series().unique()))
        self.test_turbines = np.arange(len(self.test.index.to_series().unique()))

    """
    def cluestering(self,
                    train,
                    validation,
                    test=None,
                    min_cluster_size=100):

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True).fit(
            train[['setting 1', 'setting 2', 'setting 3']])

        train_labels, strengths = hdbscan.approximate_predict(clusterer, train[['setting 1', 'setting 2', 'setting 3']])
        validation_labels, strengths = hdbscan.approximate_predict(clusterer,
                                                                   validation[['setting 1', 'setting 2', 'setting 3']])

        train['HDBScan'] = train_labels
        validation['HDBScan'] = validation_labels

        if test is not None:
            test_labels, strengths = hdbscan.approximate_predict(clusterer,
                                                                 test[['setting 1', 'setting 2', 'setting 3']])
            test['HDBScan'] = test_labels

            return train, validation, test
        else:
            return train, validation
    """

    def normalize_by_type(self,
                          train,
                          validation,
                          normalization,
                          test=None):

        df_train_type1 = train.loc[type_1]
        df_train_type2 = train.loc[type_2]

        df_validation_type1 = validation.loc[type_1]
        df_validation_type2 = validation.loc[type_2]

        df_train_type1_normalize = df_train_type1.copy()
        df_validation_type1_normalize = df_validation_type1.copy()

        if normalization == 'Standardization':
            scaler_type1 = StandardScaler().fit(df_train_type1[sensor_columns])
        elif normalization == 'MinMaxScaler':
            scaler_type1 = MinMaxScaler().fit(df_train_type1[sensor_columns])

        df_train_type1_normalize[sensor_columns] = scaler_type1.transform(df_train_type1[sensor_columns])
        df_validation_type1_normalize[sensor_columns] = scaler_type1.transform(df_validation_type1[sensor_columns])

        df_train_type1 = df_train_type1_normalize.copy()
        df_validation_type1 = df_validation_type1_normalize.copy()

        del (df_train_type1_normalize, df_validation_type1_normalize)

        df_train_type2_normalize = df_train_type2.copy()
        df_validation_type2_normalize = df_validation_type2.copy()

        gb = df_train_type2.groupby('HDBScan')[sensor_columns]

        d = {}

        for x in gb.groups:
            if normalization == 'Standardization':
                d["scaler_type2_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
            elif normalization == 'MinMaxScaler':
                d["scaler_type2_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))

            df_train_type2_normalize.loc[df_train_type2_normalize['HDBScan'] == x, sensor_columns] = d[
                "scaler_type2_{0}".format(x)].transform(
                df_train_type2.loc[df_train_type2['HDBScan'] == x, sensor_columns])
            df_validation_type2_normalize.loc[df_validation_type2_normalize['HDBScan'] == x, sensor_columns] = d[
                "scaler_type2_{0}".format(x)].transform(
                df_validation_type2.loc[df_validation_type2['HDBScan'] == x, sensor_columns])

        df_train_type2 = df_train_type2_normalize.copy()
        df_validation_type2 = df_validation_type2_normalize.copy()

        del (df_train_type2_normalize, df_validation_type2_normalize)

        df_train_all = pd.concat([df_train_type1, df_train_type2])
        df_validation_all = pd.concat([df_validation_type1, df_validation_type2])

        if test is not None:
            df_test_type1 = test.loc[type_1]
            df_test_type2 = test.loc[type_2]

            df_test_type1_normalize = df_test_type1.copy()
            df_test_type1_normalize[sensor_columns] = scaler_type1.transform(df_test_type1[sensor_columns])
            df_test_type1 = df_test_type1_normalize.copy()

            del (df_test_type1_normalize)

            df_test_type2_normalize = df_test_type2.copy()

            for x in gb.groups:
                df_test_type2_normalize.loc[df_test_type2_normalize['HDBScan'] == x, sensor_columns] = d[
                    "scaler_type2_{0}".format(x)].transform(
                    df_test_type2.loc[df_test_type2['HDBScan'] == x, sensor_columns])

            df_test_type2 = df_test_type2_normalize.copy()

            del (df_test_type2_normalize)

            df_test_all = pd.concat([df_test_type1, df_test_type2])
            df_test_all.sort_index(inplace=True)

            return df_train_all, df_validation_all, df_test_all
        else:
            return df_train_all, df_validation_all

    def binarize(self,
                 train,
                 validation,
                 test=None):

        setting_operational = ["setting_op {}".format(s) for s in range(1, 7)]
        dataset_id_columns = ["dataset_id {}".format(s) for s in range(1, 5)]

        preprocess_HDBscan = LabelBinarizer()
        preprocess_ID = LabelBinarizer()

        preprocess_HDBscan.fit(train['HDBScan'])
        preprocess_ID.fit(train.reset_index()['dataset_id'])

        dataframe_HDBscan = pd.DataFrame(preprocess_HDBscan.transform(train['HDBScan']),
                                         columns=setting_operational)
        dataframe_dataset_id = pd.DataFrame(preprocess_ID.transform(train.reset_index()['dataset_id']),
                                            columns=dataset_id_columns)

        dataframe_HDBscan_validation = pd.DataFrame(preprocess_HDBscan.transform(validation['HDBScan']),
                                                    columns=setting_operational)
        dataframe_dataset_id_validation = pd.DataFrame(
            preprocess_ID.transform(validation.reset_index()['dataset_id']), columns=dataset_id_columns)

        train = train.reset_index().join(dataframe_HDBscan)
        train = train.join(dataframe_dataset_id)

        validation = validation.reset_index().join(dataframe_HDBscan_validation)
        validation = validation.join(dataframe_dataset_id_validation)

        if test is not None:
            dataframe_HDBscan_test = pd.DataFrame(preprocess_HDBscan.transform(test['HDBScan']),
                                                  columns=setting_operational)
            dataframe_dataset_id_test = pd.DataFrame(preprocess_ID.transform(test.reset_index()['dataset_id']),
                                                     columns=dataset_id_columns)

            test = test.reset_index().join(dataframe_HDBscan_test)
            test = test.join(dataframe_dataset_id_test)

            return train, validation, test
        else:
            return train, validation

    def transform_data(self,
                       df,
                       length_sequence):

        array_data = []
        array_data_label = []

        for index_train in (df.index.to_series().unique()):
            temp_df_train = df.loc[index_train]

            for i in range(1, len(temp_df_train) + 1):
                train_x = np.ones((length_sequence, (len(temp_df_train.columns) - 1))) * -1000
                train_y = np.ones((length_sequence, 1)) * -1000

                if i - length_sequence < 0:
                    x = 0
                else:
                    x = i - length_sequence

                data = temp_df_train.iloc[x:i]

                label = data['RUL'].copy().values
                data = data.drop(['RUL'], axis=1).values
                train_x[-len(data):, :] = data

                train_y[-len(data):, 0] = label
                array_data.append(train_x)
                array_data_label.append(train_y)

        return np.array(array_data), np.array(array_data_label)

    def prepare_datareader(self,
                           batch_size,
                           validation_split,
                           number_steps_train,
                           normalization):

        train_turbines, validation_turbines = train_test_split(self.train_turbines, test_size=validation_split,
                                                               random_state=1337)

        idx_train = self.train.index.to_series().unique()[train_turbines]
        idx_validation = self.train.index.to_series().unique()[validation_turbines]
        idx_test = self.test.index.to_series().unique()[self.test_turbines]

        train = self.train.loc[idx_train]
        validation = self.train.loc[idx_validation]
        test = self.test.loc[idx_test]

        self.testII = test

        train, validation, test = self.normalize_by_type(train, validation, normalization, test)

        train, validation, test = self.binarize(train, validation, test)

        train = train.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 10', 'sensor 16', 'sensor 18', 'sensor 19'], axis=1)
        validation = validation.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 10', 'sensor 16', 'sensor 18', 'sensor 19'], axis=1)
        test = test.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 10', 'sensor 16', 'sensor 18', 'sensor 19'], axis=1)

        self.test_test = test

        self.train_data, self.train_label_data = self.transform_data(train, number_steps_train)
        self.validation_data, self.validation_label_data = self.transform_data(validation, number_steps_train)
        self.test_data, self.test_label_data = self.transform_data(test, number_steps_train)

        self.train_data, self.train_label_data = shuffle(self.train_data, self.train_label_data, random_state=1337)

        self.train_length = len(self.train_data)
        self.validation_length = len(self.validation_data)
        self.test_length = len(self.test_data)

        self.train_steps = round(len(self.train_data) / batch_size + 0.5)
        self.validation_steps = round(len(self.validation_data) / batch_size + 0.5)
        self.test_steps = round(len(self.test_data) / batch_size + 0.5)

        self.train_generator = self.generator_train(batch_size)
        self.validation_generator = self.generator_validation(batch_size)
        self.test_generator = self.generator_test(batch_size)

    def calculate_unique_turbines_cv(self,
                                     splits=5):

        cv = []
        cv_val = []

        split = KFold(n_splits=splits, shuffle=True, random_state=1338)

        for train_index, validation_index in split.split(np.arange(len(self.train.index.to_series().unique()))):
            cv.append(train_index)
            cv_val.append(validation_index)

        return cv, cv_val

    def prepare_datareader_cv(self,
                              cv_indexes,
                              cv_val_indexes,
                              batch_size,
                              number_steps_train,
                              normalization):

        idx_train = self.train.index.to_series().unique()[cv_indexes]
        idx_val = self.train.index.to_series().unique()[cv_val_indexes]

        train = self.train.loc[idx_train]
        validation = self.train.loc[idx_val]

        train, validation = self.normalize_by_type(train, validation, normalization)

        train, validation = self.binarize(train, validation)

        train = train.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 10', 'sensor 16', 'sensor 18', 'sensor 19'], axis=1)
        validation = validation.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 10', 'sensor 16', 'sensor 18', 'sensor 19'], axis=1)

        self.train_data, self.train_label_data = self.transform_data(train, number_steps_train)
        self.validation_data, self.validation_label_data = self.transform_data(validation, number_steps_train)

        self.train_data, self.train_label_data = shuffle(self.train_data, self.train_label_data, random_state=1337)

        self.train_length = len(self.train_data)
        self.validation_length = len(self.validation_data)

        self.train_steps = round(len(self.train_data) / batch_size + 0.5)
        self.validation_steps = round(len(self.validation_data) / batch_size + 0.5)

        self.train_generator = self.generator_train(batch_size)
        self.validation_generator = self.generator_validation(batch_size)

    def generator_train(self,
                        batch_size):

        while True:
            self.train_data, self.train_label_data = shuffle(self.train_data, self.train_label_data, random_state=1337)
            for ndx in range(0, self.train_length, batch_size):
                yield self.train_data[ndx:min(ndx + batch_size, self.train_length)], self.train_label_data[
                                                                                     ndx:min(ndx + batch_size,
                                                                                             self.train_length)]

    def generator_validation(self,
                             batch_size):

        while True:
            for ndx in range(0, self.validation_length, batch_size):
                yield self.validation_data[
                      ndx:min(ndx + batch_size, self.validation_length)], self.validation_label_data[
                                                                          ndx:min(ndx + batch_size,
                                                                                  self.validation_length)]

    def generator_test(self,
                       batch_size):

        while True:
            for ndx in range(0, self.test_length, batch_size):
                yield self.test_data[ndx:min(ndx + batch_size, self.test_length)], self.test_label_data[
                                                                                   ndx:min(ndx + batch_size,
                                                                                           self.test_length)]