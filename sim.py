import numpy as np
import random
from tqdm import tqdm
from scipy import stats


def create_adj_mat():
    M = np.zeros((n, n), dtype=int)
    for i in tqdm(range(n)):
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
            if random.random() > 0.5: tempA.add(elem)
            else: tempB.add(elem)
    else:
        tempA.update(joint)
    # todo add joint elements always to A when worst case?
    A.update(tempA)
    B.update(tempB)
    return A, B


def simulation(max_dif=False, simultanous=True):
    # create adj matrix
    adj_mat = create_adj_mat()
    print("finished computing adjacency matrix")
    print()

    # sample two disjoint initial sets
    if max_dif:
        A = set(range(n)[:set_size])
        B = set(range(n)[-set_size:])
    else:
        sets = random.sample(range(n), 2 * set_size)
        A = set(sets[:set_size])
        B = set(sets[set_size:])
    print("finished sampling initial sets")
    print()

    # add neutral nodes
    end = False
    counter = 0
    while not end:
        sizeA = len(A)
        sizeB = len(B)
        A, B = one_step(A, B, adj_mat, simultanous)
        if sizeA == len(A) and sizeB == len(B):
            end = True
        if len(A) + len(B) > counter + 1000:
            percentage = int((len(A) + len(B)) * 100 / n)
            print(str(percentage) + "% of nodes covered")
    print()
    print("number of nodes: " + str(n))
    if max_dif:
        print("worst case mode")
    else:
        print("random mode")
    print("initial set size: " + str(set_size))
    print("node self initiated connections: " + str(initiate_con))
    print("max connections per node: " + str(max_connections))
    _, minmax, avg_cons, var_cons, __, ___, = stats.describe(num_cons)
    print("avg connections per node: " + str(avg_cons))
    print("var of connections per node: " + str(var_cons))
    print("min connections per node: " + str(minmax[0]))
    print("max connections per node: " + str(minmax[1]))
    print()
    print("size of group A: " + str(len(A)))
    print("size of group B: " + str(len(B)))
    print("-------------------------------------------------------------")
    # print("isolated nodes: " + str(n - len(A) - len(B)))

    return



if __name__ == '__main__':
    n = 10000
    set_size = 1
    max_connections = 128
    num_cons = np.zeros(n)
    initiate_con = 16
    simulation(max_dif=False, simultanous=True)
    print("********************* started sim #2 *********************")
    # reset connection numbers
    num_cons = np.zeros(n)
    simulation(max_dif=False, simultanous=False)

