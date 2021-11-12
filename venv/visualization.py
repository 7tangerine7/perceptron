import numpy as np
from matplotlib import pyplot as plt
from numpy import arange, meshgrid, sqrt, pi, linalg, exp


def draw(x,y):
    plt.figure(figsize=(12, 7))
    plt.scatter(x[:, 0], x[:, 1], marker='o', color='blue')
    plt.scatter(y[:, 0], y[:, 1], marker='o', color='green')
    plt.show()

def draw_classes(x, y, m_t, sigma_t, theta_t):

    d = np.linspace(0, 1)
    j = np.linspace(0, 1)
    z1, z2 = np.meshgrid(d, j)
    res = 1 / (2 * pi * 1 / sqrt(linalg.det(sigma_t))) * exp(-0.5 * ((z1 - m_t[0])**2 * sigma_t[0,0] + 2 * (z1 - m_t[0]) * (z2 - m_t[1]) * sigma_t[0, 1] + (z2 - m_t[1])**2 * sigma_t[1, 1])) - theta_t

    plt.figure(figsize=(12, 7))
    plt.contour(z1, z2, res, [0])
    plt.scatter(x[:, 0], x[:, 1], marker='o', color='blue')
    plt.scatter(y[:, 0], y[:, 1], marker='o', color='green')
    plt.show()

