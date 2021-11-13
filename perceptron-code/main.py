import numpy as np
import codecs
import json
import os
from visualization import draw, draw_classes
import doctest


def get_data(file):
    """
    get data from files
    :param file: list with names of json files
    :return train_data_inside, train_data_outside, test_data:
    numpy arrays of data in first class, second class and test data
    """
    script_dir = os.path.dirname(__file__)
    file_path_1 = os.path.join(script_dir, file)
    file_path_1 = os.path.normpath(file_path_1)
    obj1 = codecs.open(file_path_1, 'r', encoding='utf-8').read()
    train_data_inside = np.array(json.loads(obj1)['inside'])  # k = 1
    train_data_outside = np.array(json.loads(obj1)['outside'])  # k = -1
    return train_data_inside, train_data_outside


def vectors_change(first_t, second_t):
    """
    :param first_t: array from 'inner' part
    :param second_t: array from 'outer' part
    :return: train_data

    >>> vectors_change(np.array([[0, 1]]), np.array([[4, -4]]))
    array([[  1.,   0.,   1.,   0.,   0.,   0.,   1.,   1.],
           [  1.,   4.,  -4.,  16., -16., -16.,  16.,  -1.]])

    """
    f = len(first_t)
    s = len(second_t)
    data = np.ones((f + s, 8))
    data[:f, 1], data[f:, 1] = first_t[:, 0], second_t[:, 0]
    data[:f, 2], data[f:, 2] = first_t[:, 1], second_t[:, 1]
    data[:f, 3], data[f:, 3] = first_t[:, 0] ** 2, second_t[:, 0] ** 2
    data[:f, 4], data[f:, 4] = first_t[:, 0] * first_t[:, 1], second_t[:, 0] * second_t[:, 1]
    data[:f, 5], data[f:, 5] = data[:f, 4], data[f:, 4]
    data[:f, 6], data[f:, 6] = first_t[:, 1] ** 2, second_t[:, 1] ** 2
    data[f:, 7] = -1  # 8th element shows class
    return data


def bounders(w, v):
    '''

    :param w:
    :param v:
    :return: matrix of vectors for adding

    >>> bounders(np.array([-1, 6]), np.array([[2, 1], [-1, 3]]))
    (array([[ 0.,  0.,  0.,  4., -2., -2.,  1.,  1.]]), 1)

    '''

    idx = np.where(w <= 0)
    a = np.zeros((len(idx[0]), 8))
    j = 0
    for i in idx[0]:
        a[j] = np.array([0, 0, 0, v[0, i]**2, v[1, i] * v[0, i], v[1, i] * v[0, i], v[1, i] ** 2, 1])
        j += 1
    return a, len(idx[0])


def parameters(temp):

    sigma11 = temp[3]
    sigma12 = temp[4]
    sigma22 = temp[6]

    sigma_t = np.array([[sigma11, sigma12], [sigma12, sigma22]])
    determinant = sigma11 * sigma22 - sigma12**2

    k = - 0.5 * temp[1:3]
    m_t = k @ np.linalg.inv(sigma_t)
    y = m_t.T @ sigma_t @ m_t
    theta_t = np.sqrt(determinant) / (2 * np.pi) * np.exp(0.5 * (temp[0] - y)) # here should be 1/det, but i have too small number
    return m_t, sigma_t, theta_t


def perceptron(data, temp):
    """
    learning parameters
    """

    data = np.vstack((data, np.zeros(8)))
    y = data[:, 7] * (data[:, :7] @ temp.T)
    num = 1
    it = 0
    while np.any(y <= 0):
        it += 1
        if num != 0:
            data = data[:-num, :]
            y = y[:-num]

        num = 0

        idx = np.where(y <= 0)
        wrong_class = data[idx[0]]

        if wrong_class.size != 0:
            temp = temp + wrong_class[0][:7] * wrong_class[0][7]
        else:
            temp += add[0][:7]

        sigma11 = temp[3]
        sigma12 = temp[4]
        sigma22 = temp[6]

        sigma_t = np.array([[sigma11, sigma12], [sigma12, sigma22]])

        values, vectors = np.linalg.eig(sigma_t)  # find vector to add
        k = 0
        if np.any(values <= 0):
            add, k = bounders(values, vectors)
            num += k
            for i in add:
                data = np.vstack((data, i))
            temp += add[0][:7]
        """
        here goes painting for every iteration
        m_t, sigma_t, theta_t = parameters(temp)
        data_draw_first_cl = data[np.where(data[:, 7] == 1)]
        data_draw_first = data_draw_first_cl[:, 1:3]
        data_draw_second_cl = data[np.where(data[:, 7] == -1)]
        data_draw_second = data_draw_second_cl[:, 1:3]
        
   #     draw_classes(data_draw_first, data_draw_second, m_t, sigma_t, theta_t)
   """
        y = data[:, 7] * (data[:, :7] @ temp.T)
    return temp


def classification(theta_t, sigma_t, m_t, data):
    """

    :param theta_t:
    :param sigma_t:
    :param m_t:
    :param data:
    :return:

    >>> classification(0.06, np.array([[1, 2], [1, 3]]), np.array([1, 1]), np.array([[2, 1], [-1, 3]]))
    array(['inside', 'outside'], dtype='<U7')
    """
    k = len(data)
    s = data - np.tile(m_t, (k, 1))
    temp = - 0.5 * s @ sigma_t @ s.T
    temp = np.diag(temp)
    result = 1 / (2 * np.pi * 1 / np.sqrt(np.linalg.det(sigma_t))) * np.exp(temp)

    classes_t = np.where(result > theta_t, "inside", "outside")
    return classes_t


def main():

    files = "test.json"
    first, second = get_data(files)
    train_data = vectors_change(second, first)  # first: -1, second: 1
    draw(second, first)

    param = perceptron(train_data, np.zeros(7))

    m, sigma, theta = parameters(param)

    mean = m[0]
    std = 1
    test_data = np.random.normal(mean, std, size=(5, 2))

    draw_classes(second, first, test_data,  m, sigma, theta)
    classes = classification(theta, sigma, m, test_data)
    print(classes)


if __name__ == '__main__':
    main()


