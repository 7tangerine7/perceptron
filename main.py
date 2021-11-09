import numpy as np
import codecs
import json
import os


def get_data(file):
    """
    get data from files
    :param file: list with names of json files
    :return train_data_inside, train_data_outside, test_data:
    numpy arrays of data in first class, second class and test data
    """
    script_dir = os.path.dirname(__file__)
    file_path_1 = os.path.join(script_dir, file[0])
    file_path_2 = os.path.join(script_dir, file[1])
    file_path_1 = os.path.normpath(file_path_1)
    file_path_2 = os.path.normpath(file_path_2)
    obj1 = codecs.open(file_path_1, 'r', encoding='utf-8').read()
    obj2 = codecs.open(file_path_2, 'r', encoding='utf-8').read()
    train_data_inside = np.array(json.loads(obj1)['inside'])  # k = 1
    train_data_outside = np.array(json.loads(obj1)['outside'])  # k = 0
    test_data_inside = np.array(json.loads(obj2)['inside'])
    test_data_outside = np.array(json.loads(obj2)['outside'])
    return train_data_inside, train_data_outside, test_data_inside, test_data_outside


def vectors_change(first, second):
    """
    :param first: array from 'inner' part
    :param second: array from 'outer' part
    :return: train_data
    """
    f = len(first)
    s = len(second)
    data = np.ones((f + s, 8))
    data[:f, 1], data[f:, 1] = first[:, 0], second[:, 0]
    data[:f, 2], data[f:, 2] = first[:, 1], second[:, 1]
    data[:f, 3], data[f:, 3] = first[:, 0] ** 2, second[:, 0] ** 2
    data[:f, 4], data[f:, 4] = first[:, 0] * first[:, 1], second[:, 0] * second[:, 1]
    data[:f, 5], data[f:, 5] = data[:f, 4], data[f:, 4]
    data[:f, 6], data[f:, 6] = first[:, 1] ** 2, second[:, 1] ** 2
    data[f:, 7] = 0  # 8th element shows class
    return data


def perceptron(data):
    """
    learning parameters
    """
    temp = np.zeros(7)  # vector lambda
    y = data[:, 7] * (data[:, :7] @ temp.T)
    while y.any() <= 0:
        idx = np.where(y == 0)
        wrong_class = data[idx]
        temp += wrong_class[0, 7] * wrong_class[0, :7]
        y = data[:, 7] * (data[:, :7] @ temp.T)
    return temp


if __name__ == '__main__':
    files = ['train_01.json', 'train_02.json']
    train_first, train_second, test_first, test_second = get_data(files)
    train_data = vectors_change(train_first, train_second)
    param = perceptron(train_data)
    print(param)




