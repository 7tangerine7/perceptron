import numpy as np
import codecs
import json
import os


def getData(file):
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


if __name__ == '__main__':
    files = ['train_01.json', 'train_02.json']
    train_first, train_second, test_first, test_second = getData(files)
    print(train_first)


