import numpy as np
import random
from copy import deepcopy
import pylab as pl
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def create_adj_mat(n):
    num_cons = np.zeros(n)
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

    # print("num_cons:" + str(stats.describe(num_cons)))

    # plt.hist(num_cons, 128, density=True, facecolor='g', alpha=0.75)
    # plt.xlabel('number of connections')
    # plt.ylabel('Probability')
    # plt.title('Histogram of Connections number')
    # # plt.xlim(0, 136)
    # plt.grid(True)
    # plt.show()
    return M


def one_step(n: int, sets: list, adj_mat: np.array, simultanous: bool):
    sets_num = len(sets)
    new_nodes_dict = dict()
    all_added_nodes = []
    for s in sets:
        all_added_nodes += s
    for i in range(n):  # sort all nodes to their set
        if i not in all_added_nodes:  # node not in a set already
            i_neighbors = np.nonzero(adj_mat[i])
            i_sets = []
            for s in range(sets_num):
                is_neighbors = len(np.intersect1d(sets[s], i_neighbors))
                i_sets += [s]*is_neighbors  # add s idx as the number of neighbors i has in s
            if len(i_sets) > 0:
                new_nodes_dict[i] = i_sets
    # add nodes by chances
    for node in new_nodes_dict:
        chosen = random.choice(new_nodes_dict[node])
        sets[chosen].append(node)

    return sets










    # -----------------------------------------------------
    # tempA = set()
    # tempB = set()
    # for node in A:
    #     curA = set(np.argwhere(adj_mat[node] == 1).flatten())
    #     curA = curA - B
    #     tempA.update(curA)
    # for node in B:
    #     curB = set(np.argwhere(adj_mat[node] == 1).flatten())
    #     curB = curB - A
    #     tempB.update(curB)
    # joint = tempA.intersection(tempB)
    # tempA = tempA - joint
    # tempB = tempB - joint
    #
    # # for each joint element choose by coin flip
    # if simultanous:
    #     for elem in joint:
    #         neighbors = set(np.argwhere(adj_mat[elem] == 1).flatten())
    #         a_neighbors = len(A.intersection(neighbors))
    #         b_neighbors = len(B.intersection(neighbors))
    #         if random.random() < a_neighbors / (a_neighbors + b_neighbors):
    #             tempA.add(elem)
    #         else:
    #             tempB.add(elem)
    # else:
    #     tempA.update(joint)
    # # todo add joint elements always to A when worst case?
    #
    # A.update(tempA)
    # B.update(tempB)
    # return A, B


def simulation(n, init_set_sizes, max_dif=False, simultanous=True):
    # if max_dif:
    #     print("worst case mode")
    # else:
    #     print("random mode")
    sets_num = len(init_set_sizes)
    # create adj matrix
    adj_mat = create_adj_mat(n)
    # print("finished computing adjacency matrix")
    # print()
    sets = []
    # sample two disjoint initial sets
    if sets_num == 2 and max_dif:
        sets.append(range(n)[:init_set_sizes[0]])
        sets.append(range(n)[-init_set_sizes[1]:])
    else:
        samples = random.sample(range(n), int(np.sum(init_set_sizes)))
        for i, s in enumerate(init_set_sizes):
            prev_nodes = int(np.sum(init_set_sizes[:i]))
            cur_set = samples[prev_nodes:prev_nodes + s]
            sets.append(cur_set)


    # print("finished sampling initial sets")
    # print()

    # add neutral nodes
    nafot = np.array(init_set_sizes).reshape(-1, 1)
    end = False
    # print(sets)
    while not end:
        sets = one_step(n, sets, adj_mat, simultanous)
        cur_added_nodes = 0   # current number of nodes with some tx
        for i in range(sets_num):
            cur_added_nodes += len(sets[i])
        if cur_added_nodes == n:  # all nodes added
            end = True
        else:
            for i in range(sets_num):
                np.append(nafot[i], len(sets[i]))
        # print(cur_added_nodes)

    end_sizes = []
    for i in range(sets_num):
        end_sizes.append(len(sets[i]))
        # if len(A) + len(B) > counter + 1:
        #     percentage = int((len(A) + len(B)) * 100 / n)
        #     print(str(percentage) + "% of nodes covered")
    # plot growth rate of each group todo change plots to nafot format


    # plt.plot(cuma, label="A")
    # plt.plot(cumb, label="B")
    # plt.xlabel('t')
    # plt.ylabel('num. of nodes')
    # plt.title('# of new nodes in each time step')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # print("num_cons:" + str(stats.describe(num_cons)))

    # print()
    # print("size of group A: " + str(len(A)))
    # print("size of group B: " + str(len(B)))
    # print("-------------------------------------------------------------")
    # print(end_sizes)
    end_sizes = np.array(end_sizes)
    return end_sizes.reshape((3,1))


def single_tx_propagation(n, fit_func=None, mode=None):
    adj_mat = create_adj_mat(n)
    if mode == 'best':
        starting_nodes = range(n)[:set_size]
    elif mode == 'worst':
        starting_nodes = range(n)[-set_size:]
    else:
        starting_nodes = random.sample(range(n), set_size)
    A = set(starting_nodes)
    # add neutral nodes
    end = False
    # counter = 0
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

    print()
    print("-------------------------------------------------------------")


def logistic_growth(t, r):  # todo fit also c?
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
    # a_size = 3
    # b_size = 1
    max_connections = 128
    initiate_con = 8  # note that increasing increases nodes var but results seem similar
    print()
    print("number of nodes: " + str(n))
    print("initial single tx set size: " + str(set_size))
    # print("initial A set size: " + str(a_size))
    # print("initial B set size: " + str(b_size))
    print("node self initiated connections: " + str(initiate_con))
    print("max connections per node: " + str(max_connections))
    # simulation(max_dif=False, simultanous=True)
    # print("********************* started sim #2 *********************")
    # reset connection numbers
    # a_data = []
    # b_data = []
    # for i in tqdm(range(5000)):
    #     a, b = simulation(max_dif=False, simultanous=True)
    #     a_data.append(a)
    #     b_data.append(b)
    # print(stats.describe(a_data))
    # print(stats.describe(b_data))
    # for i in range(5):
    #     num_cons = np.zeros(n)
    #     single_tx_propagation(logistic_growth, "best")
    #     num_cons = np.zeros(n)
    #     single_tx_propagation(logistic_growth, 'best')
    results = []
    for i in tqdm(range(10)):
        s = simulation(10000, [30, 60, 120])
        if i == 0:
            results = s
        else:
            results = np.hstack((results, s))

    print("description of all results:")
    print(stats.describe(results.flatten()))
    print()
    print('----------------------------------------------------------------------')
    print('description for each tx:')
    for i, tx in enumerate(results):
        print('tx ' + str(i + 1))
        print(stats.describe(tx))






