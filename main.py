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


def bounders(w, v):
    idx = np.where(w <= 0)
    a = np.array([])
    for i in idx[0]:
        a = np.append(a, [0, 0, 0, v[i, 0]**2, v[i, 1] * v[i, 0], v[i, 1] * v[i, 0], v[i, 1] ** 2, 1])
    return a


def parameters(temp):

    sigma11 = temp[3]
    sigma12 = temp[4]
    sigma22 = temp[6]

    sigma_t = np.array([[sigma11, sigma12], [sigma12, sigma22]])
    determinant = sigma11 * sigma22 - sigma12**2

    m1 = (sigma12 * temp[2] - sigma22 * temp[1]) / determinant
    m2 = - (temp[2] + 2 * sigma11 * (sigma12 * temp[2] - sigma12 * temp[1])) / (2 * sigma22 * determinant)
    m_t = np.array([m1, m2])

    theta_t = np.exp(0.5 * (temp[0] + 1 / np.sqrt(determinant) - m_t.T @ sigma_t @ m_t))
    return m_t, sigma_t, theta_t


def perceptron(data, temp):
    """
    learning parameters
    """
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

    param = np.zeros(7)  #plak-plak
    param = perceptron(train_data, param)
    m, sigma, theta = parameters(param)
    values, vectors = np.linalg.eig(sigma)
    train_data = np.vstack((train_data, bounders(values, vectors)))  # find vector to add
    param = perceptron(train_data, param)  # add vector for correct work
    print(param)



