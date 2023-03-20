from typing import List
from inspect import isfunction
import operator

import pandas as pd
import numpy as np
from sklearn import preprocessing
from category_encoders.target_encoder import TargetEncoder
from category_encoders import OneHotEncoder, CatBoostEncoder


def cartesian_product_basic(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    返回笛卡尔积
    :param left: 原始数据集
    :param right: 扩展的列
    :return: 扩展后的数据集
    """
    return left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1)


def handle_outliers_by_quantile(data: pd.DataFrame, col_name: str, upper_percent: float, lower_percent: float,
                                is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    覆盖过大或过小的值
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param upper_percent: 上限
    :param lower_percent: 下限
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()

    upper_lim = tmp_data.loc[(tmp_data[is_train_name] == 1), col_name].quantile(upper_percent)  # 这里只用训练数据
    lower_lim = tmp_data.loc[(tmp_data[is_train_name] == 1), col_name].quantile(lower_percent)
    tmp_data.loc[(tmp_data[col_name] > upper_lim), col_name] = upper_lim
    tmp_data.loc[(tmp_data[col_name] < lower_lim), col_name] = lower_lim
    return tmp_data[col_name]


def category_to_num(data: pd.DataFrame, col_name: str, is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    类别变量简单转换为整数
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    col_type_dic = {label: idx for idx, label in
                    enumerate(np.unique(tmp_data.loc[(tmp_data[is_train_name] == 1), col_name]))}
    tmp_data[col_name] = tmp_data[col_name].map(col_type_dic)  # 这里没有考虑key不存在的情况，默认NaN，可以改成其他的
    return tmp_data[col_name].fillna(-1).astype(int)  # 为了转成int，这里填充了-1，可以改成其他的


def category_to_num_by_label_encoder(data: pd.DataFrame, col_name: str,
                                     is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    使用sklearn LabelEncoder将类别变量转为数字
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(tmp_data.loc[(tmp_data[is_train_name] == 1), col_name])
    test_tmp_data = tmp_data.loc[(tmp_data[is_train_name] == 0), col_name]
    test_tmp_data = test_tmp_data.map(lambda x: '<unknown>' if x not in encoder.classes_ else x)  # 处理test标签不在train中的情况
    tmp_data.loc[(tmp_data[is_train_name] == 0), col_name] = test_tmp_data
    encoder.classes_ = np.append(encoder.classes_, '<unknown>')
    return encoder.transform(tmp_data[col_name])


def category_encoding_by_onehot(data: pd.DataFrame, col_name: str, is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    类别变量onehot化
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, is_train_name]].copy()
    encoder = OneHotEncoder(cols=[col_name], handle_unknown='indicator', handle_missing='indicator', use_cat_names=True)
    encoder.fit(tmp_data.loc[(tmp_data[is_train_name] == 1), col_name])
    return encoder.transform(tmp_data[col_name])


def category_encoding_by_target_encoder(data: pd.DataFrame, col_name: str, label_name: str = 'LABEL',
                                        is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    类别变量target encoding处理
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param label_name: 标签类名
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, label_name, is_train_name]].copy()
    encoder = TargetEncoder(cols=[col_name], handle_unknown='value', handle_missing='value') \
        .fit(tmp_data.loc[(tmp_data[is_train_name] == 1), col_name],
             tmp_data.loc[(tmp_data[is_train_name] == 1), label_name])  # 在训练集上训练
    return encoder.transform(tmp_data[col_name])


def category_encoding_by_catboost(data: pd.DataFrame, col_name: str, label_name: str = 'LABEL',
                                  is_train_name: str = 'IS_TRAIN') -> pd.DataFrame:
    """
    类别变量catboost encoding处理
    :param data: 原始数据集
    :param col_name: 列名，这里只支持单个变量
    :param label_name: 标签类名
    :param is_train_name: 用来筛选训练数据的列名
    :return: 只返回处理的单列
    """
    tmp_data = data[[col_name, label_name, is_train_name]].copy()
    encoder = CatBoostEncoder(cols=[col_name], handle_unknown='value', handle_missing='value') \
        .fit(tmp_data.loc[(tmp_data[is_train_name] == 1), col_name],
             tmp_data.loc[(tmp_data[is_train_name] == 1), label_name])  # 在训练集上训练
    return encoder.transform(tmp_data[col_name])


def binary_cross_columns(data: pd.DataFrame, cross_cols: List[List]) -> pd.DataFrame:
    """
    类别变量交叉组合
    :param data: 原始数据集
    :param cross_cols: like [['fea1', 'fea2'], ['fea3', 'fea4']], 类型要求是objects(str)
    :return: 生成的新的特征集
    """
    all_cols = set([cols for cross_item in cross_cols for cols in cross_item])
    tmp_data = data[all_cols].copy()

    col_names = ['_'.join(cross_item) for cross_item in cross_cols]
    cross_dict = {k: v for k, v in zip(col_names, cross_cols)}

    for k, v in cross_dict.items():
        tmp_data[k] = tmp_data[v].apply(lambda x: '-'.join(x), axis=1)

    return tmp_data[col_names]
