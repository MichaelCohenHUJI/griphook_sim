import numpy as np
import random

import pylab as pl
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def create_adj_mat():
    M = np.zeros((n, n), dtype='uint8')
    for i in range(n):
        found_cons = False if i > initiate_con - 1 else True
        temp = np.arange(i)
        while not found_cons: # find 8 connection
            temp = random.sample(range(i), initiate_con)
            # check no self connection and no saturated neighbor
            found_cons = True
            for j in temp:
                if j == i:
                    found_cons = False
                    break
                elif num_cons[j] >= max_connections:
                    found_cons = False
                    break
        # update num connection and adj matrix
        for l in temp:
            M[i, l] = 1
            M[l, i] = 1
            num_cons[l] += 1
            num_cons[i] += 1
    # plt.hist(num_cons, 128, density=True, facecolor='g', alpha=0.75)
    # plt.xlabel('number of connections')
    # plt.ylabel('Probability')
    # plt.title('Histogram of Connections number')
    # # plt.xlim(0, 136)
    # plt.grid(True)
    # plt.show()
    return M


def one_step(A: set, B: set, adj_mat: np.array, simultanous: bool):
    tempA = set()
    tempB = set()
    for node in A:
        curA = set(np.argwhere(adj_mat[node] == 1).flatten())
        curA = curA - B
        tempA.update(curA)
    for node in B:
        curB = set(np.argwhere(adj_mat[node] == 1).flatten())
        curB = curB - A
        tempB.update(curB)
    joint = tempA.intersection(tempB)
    tempA = tempA - joint
    tempB = tempB - joint

    # for each joint element choose by coin flip
    if simultanous:
        for elem in joint:
            neighbors = set(np.argwhere(adj_mat[elem] == 1).flatten())
            a_neighbors = len(A.intersection(neighbors))
            b_neighbors = len(B.intersection(neighbors))
            if random.random() < a_neighbors / (a_neighbors + b_neighbors):
                tempA.add(elem)
            else:
                tempB.add(elem)
    else:
        tempA.update(joint)
    # todo add joint elements always to A when worst case?

    A.update(tempA)
    B.update(tempB)
    return A, B


def simulation(max_dif=False, simultanous=True):
    # if max_dif:
    #     print("worst case mode")
    # else:
    #     print("random mode")

    # create adj matrix
    adj_mat = create_adj_mat()
    # print("finished computing adjacency matrix")
    # print()

    # sample two disjoint initial sets
    if max_dif:
        A = set(range(n)[:a_size])
        B = set(range(n)[-b_size:])
    else:
        sets = random.sample(range(n), a_size + b_size)
        A = set(sets[:a_size])
        B = set(sets[-b_size:])
    # print("finished sampling initial sets")
    # print()

    # add neutral nodes
    ansfota = [a_size]
    ansfotb = [b_size]
    end = False
    # counter = 0
    while not end:
        sizeA = len(A)
        sizeB = len(B)
        A, B = one_step(A, B, adj_mat, simultanous)
        if sizeA == len(A) and sizeB == len(B):
            end = True
        else:
            ansfota.append(len(A) - sizeA)
            ansfotb.append(len(B) - sizeB)
        # if len(A) + len(B) > counter + 1:
        #     percentage = int((len(A) + len(B)) * 100 / n)
        #     print(str(percentage) + "% of nodes covered")
    # plot growth rate of each group
    cuma = np.cumsum(ansfota)
    cumb = np.cumsum(ansfotb)
    # plt.plot(ansfota, label="A")
    # plt.plot(ansfotb, label="B")
    # plt.xlabel('t')
    # plt.ylabel('added nodes')
    # plt.title('# of new nodes in each time step')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plot number of nodes in each group
    # a = np.insert(cuma[:-1], 0, 0)

    # plt.plot(cuma, label="A")
    # plt.plot(cumb, label="B")
    # plt.xlabel('t')
    # plt.ylabel('num. of nodes')
    # plt.title('# of new nodes in each time step')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # _, minmax, avg_cons, var_cons, __, ___, = stats.describe(num_cons)
    # print("avg connections per node: " + str(avg_cons))
    # print("var of connections per node: " + str(var_cons))
    # print("min connections per node: " + str(minmax[0]))
    # print("max connections per node: " + str(minmax[1]))
    # print()
    # print("size of group A: " + str(len(A)))
    # print("size of group B: " + str(len(B)))
    # print("-------------------------------------------------------------")
    # # print("isolated nodes: " + str(n - len(A) - len(B)))
    return len(A), len(B)


def single_tx_propagation(fit_func=None, mode=None):
    adj_mat = create_adj_mat()
    if mode == 'best':
        starting_nodes = range(n)[:set_size]
    elif mode == 'worst':
        starting_nodes = range(n)[-set_size:]
    else:
        starting_nodes = random.sample(range(n), set_size)
    A = set(starting_nodes)
    # add neutral nodes
    end = False
    counter = 0
    added_as_func_of_time = [set_size]
    while not end:
        sizeA = len(A)
        tempA = set()
        for node in A:
            curA = set(np.argwhere(adj_mat[node] == 1).flatten())
            tempA.update(curA)
        A.update(tempA)
        if sizeA == len(A):
            end = True
        else:
            added_as_func_of_time.append(len(A) - sizeA)
        # if len(A) > counter + 1000:
        #     percentage = int(len(A) * 100 / n)
        #     print(str(percentage) + "% of nodes covered")
    nodes_as_func_of_time = np.cumsum(added_as_func_of_time)
    print(nodes_as_func_of_time)
    if fit_func is not None:
        global n0
        n0 = set_size
        fit_data(fit_func, range(len(nodes_as_func_of_time)), nodes_as_func_of_time)
        return

    plt.plot(nodes_as_func_of_time)
    plt.xlabel('t')
    plt.ylabel('total nodes')
    # plt.title('# of new nodes in each time step')
    plt.title('total nodes')
    plt.legend()
    plt.grid(True)
    plt.show()


    _, minmax, avg_cons, var_cons, __, ___, = stats.describe(num_cons)
    print("avg connections per node: " + str(avg_cons))
    print("var of connections per node: " + str(var_cons))
    print("min connections per node: " + str(minmax[0]))
    print("max connections per node: " + str(minmax[1]))
    print()
    print("-------------------------------------------------------------")


def logistic_growth(t, r):
    c = (n - n0) / n0
    return n / (1 + c * np.exp(-r*t))


def fit_data(func, t, data):
    popt, pcov = curve_fit(func, t, data)
    plt.plot(data, label="data")
    fitted_values = logistic_growth(t, *popt)
    pl.plot(t, fitted_values, 'r-.', label='fit: r=%5.3f' % tuple(popt))
    # rplus = popt[0] + np.sqrt(np.diag(pcov))
    # rminus = popt[0] - np.sqrt(np.diag(pcov))
    # plt.plot(logistic_growth(t, rplus), linestyle=':')
    # plt.plot(logistic_growth(t, rminus), linestyle=':')
    plt.xlabel('t')
    plt.ylabel('total nodes')
    # plt.title('# of new nodes in each time step')
    plt.title('total nodes as func of time')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    n = 10
    n0 = 0
    set_size = 1
    a_size = 3
    b_size = 1
    max_connections = 128
    num_cons = np.zeros(n)
    initiate_con = 8  # note that increasing increases nodes var but results seem similar
    print()
    print("number of nodes: " + str(n))
    print("initial single tx set size: " + str(set_size))
    print("initial A set size: " + str(a_size))
    print("initial B set size: " + str(b_size))
    print("node self initiated connections: " + str(initiate_con))
    print("max connections per node: " + str(max_connections))
    # simulation(max_dif=False, simultanous=True)
    # print("********************* started sim #2 *********************")
    # reset connection numbers
    a_data = []
    b_data = []
    for i in tqdm(range(5000)):
        num_cons = np.zeros(n)
        a, b = simulation(max_dif=False, simultanous=True)
        a_data.append(a)
        b_data.append(b)
    # for i in range(5):
    #     num_cons = np.zeros(n)
    #     single_tx_propagation(logistic_growth, "best")
    #     num_cons = np.zeros(n)
    #     single_tx_propagation(logistic_growth, 'best')
    print(stats.describe(a_data))
    print(stats.describe(b_data))




