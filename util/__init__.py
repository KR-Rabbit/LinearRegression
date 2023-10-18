import numpy as np


def std_normalization(data):
    """
    标准化
    :param data: 数据
    :return: 标准化后的数据
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
