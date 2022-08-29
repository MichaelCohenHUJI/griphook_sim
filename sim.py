import numpy as np
import random
import tqdm
from scipy import stats

n = 1000
gs = 50
max_connections = 128
num_cons = np.zeros(1000)
initiate_con = 8

def create_adj_mat():
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        arr = M[i]
        found_cons = False if i > initiate_con - 1 else True
        temp = np.arange(i)
        while not found_cons: # find 8 connection
            temp = random.sample(range(n), 8)
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




def simulation():
    adj_mat = create_adj_mat()




A = create_adj_mat()

print(1)