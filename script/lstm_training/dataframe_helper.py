#!/usr/bin/env python

import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import rospy
import pandas as pd
from param_loader import ParamLoader


class DataframePreparator():
    def __init__(self,par: ParamLoader):
        self.pl = par
        self.df_array = self.import_csv()
        self.n_samples = np.empty(self.pl.n_of_trials)

        if self.pl.use_scaler_150rec:
            self.array_of_scaler = joblib.load(self.pl.data_path + '/array_of_scalers150.gz')
        else:
            self.array_of_scaler = self.fit_scaler()
            joblib.dump(self.array_of_scaler, self.pl.data_path + '/array_of_scalers.gz')

    def fit_scaler(self):
        head = np.array(self.pl.headers)
        array_of_scaler = np.empty_like(head, dtype=object)

        for i in np.arange(head.shape[0]):  # iterate over topics
            for j in np.arange(head.shape[1]):  # iterate over headers
                array_of_scaler[i, j] = MinMaxScaler(feature_range=(-1, 1))
                # tmp array of 'x' topic, 'y' headers of all the records
                tmp_array = np.array([])
                for k in np.arange(self.pl.n_of_trials):
                    tmp_array = np.concatenate((tmp_array, self.df_array[k, i][head[i, j]]), axis=0)
                array_of_scaler[i, j].fit(tmp_array.reshape(-1, 1))

        return array_of_scaler

    def import_csv(self):
        df_array = np.empty([self.pl.n_of_trials, len(self.pl.topics)], dtype=object)

        for n in np.arange(self.pl.n_of_trials):
            for j, t in enumerate(self.pl.topics):
                datapath = self.pl.data_path + '/trial_' + str(n + 1) + '/' + t[1:] + '.csv'
                df_array[n, j] = pd.read_csv(datapath)
        return df_array

    def drop_headers(self):
        head = np.array(self.pl.headers)

        for k in np.arange(self.pl.n_of_trials):
            for i in np.arange(head.shape[0]):  # iterate over topics
                col_list = list(self.df_array[k, i].columns.values)
                col_list.remove(self.pl.headers[i][0])
                col_list.remove(self.pl.headers[i][1])
                col_list.remove('Time')
                self.df_array[k, i].drop(columns=col_list, inplace=True)

    def normalizer(self):
        head = np.array(self.pl.headers)

        for k in np.arange(self.pl.n_of_trials):
            for i in np.arange(head.shape[0]):
                for j in np.arange(head.shape[1]):
                    self.df_array[k, i][head[i, j]] = \
                        self.array_of_scaler[i, j].transform(self.df_array[k, i][head[i, j]].values.reshape(-1, 1))

    def align_topics(self):

        # interpolate the wrench_topic
        for n in np.arange(self.pl.n_of_trials):
            for j in np.arange(start=1, stop=len(self.pl.topics), step=1):
                df = self.df_array[n, j]
                index_list = self.df_array[n, 0].loc[:, 'Time'].values
                self.n_samples[n] = len(index_list) - 20
                df.set_index('Time', inplace=True)
                df_interp = df.reindex(df.index.union(index_list))
                df_interp.interpolate('index', inplace=True)
                self.df_array[n, j] = df_interp.loc[list(index_list)]
                self.df_array[n, j].reset_index(inplace=True)

        # delete the first sample since is full of 'nan' after the interpolation
        for n in np.arange(self.pl.n_of_trials):
            for j in np.arange(start=0, stop=len(self.pl.topics), step=1):
                self.df_array[n, j].drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], inplace=True)
                self.df_array[n, j].reset_index(inplace=True, drop=True)

        self.n_samples.astype(int)


    def add_n_samples(self):
        head = np.array(self.pl.headers)
        for k in np.arange(self.pl.n_of_trials):
            for i in np.arange(head.shape[0]):
                n_elem = self.df_array[k, i].shape[0]
                self.df_array[k, i]["length"] = np.full(n_elem, n_elem)

    def to_tensor(self):

        data_tens = torch.zeros(self.pl.n_of_trials, self.pl.headers.shape[0], len(self.df_array[0, 0].columns), np.amax(self.n_samples).astype(int))
        # Create small tensor, fill in data_tens
        for k in np.arange(self.pl.n_of_trials):
            for i in np.arange(self.pl.headers.shape[0]):
                tmp_array_debug = self.df_array[k, i].values
                tmp_tens = torch.tensor(tmp_array_debug)
                data_tens[k, i, :, :int(self.n_samples[k])] = tmp_tens.transpose(0,1)
