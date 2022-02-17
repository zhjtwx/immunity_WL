from __future__ import print_function
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
# def load_feature():
#     target = pd.read_csv('/Users/deepwise/Documents/wf/project/xiehe/mianyi/zuexue_实验/single_max/feature/targets.csv')
#     mask_taeget = np.array(target['mask'])
#     label_taeget = np.array(target['label'])
#
#     tags = pd.read_csv('/Users/deepwise/Documents/wf/project/xiehe/mianyi/zuexue_实验/single_max/feature/tags.csv')
#     mask_tags = np.array(tags['mask'])
#     label_tags = np.array(tags['dataset'])
#
#     features = pd.read_csv('/Users/deepwise/Documents/wf/project/xiehe/mianyi/data/single_feature/single_feature.csv')
#     mask_features = np.array(features['mask'])
#     feature = np.array(features)[:, 2:]
#
#     image = np.array(features['image'])
#
#     image = (set(image))
#
#     data = [['pid', 'tags', 'label']]
#     for file in image:
#         pid = file.split('/')[-3]
#         a = [pid]
#         for i in range(len(mask_tags)):
#             if pid in mask_tags[i]:
#                 a.append(label_tags[i])
#                 break
#         for j in range(len(mask_taeget)):
#             if pid in mask_taeget[i]:
#                 a.append(label_taeget[i])
#                 break
#         if len(a) < 3 or a in data:
#             continue
#         data.append(a)
#     data = pd.DataFrame(data)
#     data.to_csv('/Users/deepwise/PycharmProjects/twx_pywk/xiehe/attention_MIL_ML/train_test_data.csv', index=False, header=False)


import math


def preprocessing(df):
    """
    预处理，去除每一列的空值，并将非数值转化为数值型数据，分两步
    1. 如果本列含有null。
        - 如果是number类型
            如果全为空，则均置零；
            否则，空值的地方取全列平均值。
        - 如果不是number类型
            将空值置NA
    2. 如果本列不是数值型数据，则用label encoder转化为数值型
    :param df: dataframe
    :return: 处理后的dataframe
    """

    def process(c):
        if c.isnull().any().any():
            if np.issubdtype(c.dtype, np.number):
                new_c = c.fillna(c.mean())
                if new_c.isnull().any().any():
                    return pd.Series(np.zeros(c.size))
                return new_c
            else:
                return pd.Series(LabelEncoder().fit_transform(c.fillna("NA").values))
        else:
            if not np.issubdtype(c.dtype, np.number):
                return pd.Series(LabelEncoder().fit_transform(c.values))
        return c

    pre_df = df.copy()
    return pre_df.apply(lambda col: process(col))


def scale_on_feature(data):
    """
    对每一列feature进行归一化，使方差一样

    :param data: dataframe
    :return: 归一化后的dataframe
    """
    data_scale = data.copy()
    data_scale[data.columns] = scale(data_scale)
    return data_scale


def scale_on_min_max(data, feature_range=(0, 1)):
    """
    对每一列feature进行相同区间归一化，使方差一样

    :param data:
    :param feature_range: dataframe
    :return: 归一化后的dataframe
    """
    data_scale = data.copy()
    scaler = MinMaxScaler(feature_range=feature_range)
    data_scale[data.columns] = scaler.fit_transform(data_scale)
    return data_scale


# 均值
features = pd.read_csv('/Users/deepwise/Documents/wf/project/xiehe/mianyi/data/single_feature/single_feature.csv')
mask_features = np.array(features['mask'])
features = preprocessing(features)
features = scale_on_min_max(features)
feature = np.array(features)[:, 2:]


# feature = normalize(feature)


def data_depent():
    tarin = []
    test = []
    info = np.array(pd.read_csv('/xiehe/attention_MIL_ML/train_test_data.csv'))
    for data in info:
        if data[1] == 0:
            if data[-1] == 0:
                tarin.append([data[0], [1, 0]])
            if data[-1] == 1:
                tarin.append([data[0], [0, 1]])
        if data[1] == 1:
            if data[-1] == 0:
                test.append([data[0], [1, 0]])
            if data[-1] == 1:
                test.append([data[0], [0, 1]])
    return tarin, test


def load_feature(pid):
    fea = np.zeros((6, 1454))
    flag = []

    flg = ''

    for i in range(len(mask_features)):
        if str(pid) in mask_features[i]:
            if flg == '':
                flg = mask_features[i][:-9]
                data = per_data(feature[i])
                flag.append(data)

                continue
            if len(flg) > 2:
                if mask_features[i][:-9] != flg:
                    flg = ''
                    break
                if mask_features[i][:-9] == flg:
                    data = per_data(feature[i])
                    flag.append(data)

    fea[:len(flag)] = np.array(flag)
    return fea


def per_data(data):
    for i in range(len(data)):
        if data[i] != data[i]:
            data[i] = 0.
    return data


class MyDataset(Dataset):
    def __init__(self, data, transform=False):
        self.image_files = data
        self.transform = transform

    def __getitem__(self, index):  # 返回的是tensor
        fea = load_feature(self.image_files[index][0])
        label = self.image_files[index][1]
        return torch.FloatTensor(fea), torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_files)
