import random
import math
import matplotlib.pyplot as plt
import numpy as np


def min_one(x):
    return sum(x)


def sphere(x, convert=None):
    if convert is None:
        return sum(x ** 2)
    else:
        return sum(convert(x) ** 2)


def hub_location_allocation(select):
    select = np.array(select)
    model = np.load('hub_model.npz')
    xc = model['xc']
    yc = model['yc']
    d = model['d']
    xs = model['xs']
    ys = model['ys']
    c = model['c']
    length_matrix = model['length_matrix']
    c_num = len(xc)
    s_num = len(xs)
    z1 = 0
    for i in range(c_num):
        j = np.argmin(length_matrix[i] / select)
        z1 += d[i] * length_matrix[i][j]

    z2 = sum(select * c)
    w1 = 1
    w2 = 1
    cost = w1 * z1 + w2 * z2
    return cost


def show_hub(select):
    # plt.close()
    select = np.array(select)
    model = np.load('hub_model.npz')
    xc = model['xc']
    yc = model['yc']
    xs = model['xs']
    ys = model['ys']
    length_matrix = model['length_matrix']
    c_num = len(xc)
    for i in range(c_num):
        j = np.argmin(length_matrix[i] / select)
        plt.plot([xc[i], xs[j]], [yc[i], ys[j]])
    plt.plot(xc, yc, 'bo')
    plt.plot(xs[select == 1], ys[select == 1], 'rs')
    plt.plot(xs[select == 0], ys[select == 0], 'rs', markerfacecolor='white')
    # plt.show(block=False)
    plt.show()



def get_distance(p1, p2):
    z = p1 - p2
    return math.sqrt(z[0] ** 2 + z[1] ** 2)


def create_model():
    cnum = 40  # client number
    snum = 20  # server number
    x = np.random.uniform(0, 99, cnum + snum)
    y = np.random.uniform(0, 99, cnum + snum)
    sc = np.arange(cnum + snum)
    ci = random.sample(list(sc), cnum)
    si = np.setdiff1d(sc, ci)

    xc = x[ci]
    yc = y[ci]
    d = np.random.randint(1, 99, cnum)  # demand for each client

    xs = x[si]
    ys = y[si]
    c = np.random.randint(8000, 12000, snum)  # installation cost for each server

    length_matrix = np.zeros((cnum, snum))
    for i in range(cnum):
        for j in range(snum):
            cpoint = np.array([xc[i], yc[i]])
            spoint = np.array([xs[j], ys[j]])
            length_matrix[i][j] = get_distance(cpoint, spoint)
    np.savez('hub_model', xc=xc, yc=yc, d=d, xs=xs, ys=ys, c=c, length_matrix=length_matrix)

# test
# hub_location_allocation(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]))
# show_hub([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])

# print(show_hub([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]))
