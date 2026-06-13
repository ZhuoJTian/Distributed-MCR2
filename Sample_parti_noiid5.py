#######################################################################################################
# For the whole system, there are 10 clients in total, this file was generated for deciding local data
# distribution for each client (non i.i.d. distribution)
#######################################################################################################

import numpy as np
import random
import sys
import json


def getAttr(NUM_OF_CLIENTS, Num_Categories):
    # Define minimum and maximum samples for each client

    RemainedSamples_train = [5000 for i in range(10)]
    RemainedSamples_test = [800 for i in range(10)]

    # 决定每一个用户拥有的样本类别数
    NumOfCategories = [Num_Categories for i in range(NUM_OF_CLIENTS)]
    '''
    # 决定每个用户拥有的具体类别
    CategoryToClients = [[] for _ in range(NUM_OF_CLIENTS)]
    for client in range(NUM_OF_CLIENTS):
        Categories = np.random.choice(10, NumOfCategories[client], replace=False)
        for item in Categories:
            CategoryToClients[client].append(item)

    # print(CategoryToClients)

    np.savetxt("./data/CategoryToClients5_4.txt", CategoryToClients, fmt='%d')
    '''
    CategoryToClients_array=np.loadtxt("./data/CategoryToClients4.txt")
    CategoryToClients =[[int(CategoryToClients_array[i, j])
                        for j in range(CategoryToClients_array.shape[1])]
                        for i in range(CategoryToClients_array.shape[0])]
    
    # 统计每一种类别有多少用户有
    Categories_client = [[] for i in range(10)]
    Categories = [0 for i in range(10)]
    for client in range(NUM_OF_CLIENTS):
        for item in CategoryToClients[client]:
            Categories_client[item].append(client)
            Categories[item]+=1
    '''
    print(Categories)

    count = Categories.copy()
    LocalDist_test = [[0.0 for _ in range(10)] for __ in range(NUM_OF_CLIENTS)]
    for client in range(NUM_OF_CLIENTS):
        for cls in CategoryToClients[client]:
            LocalDist_test[client][cls] = RemainedSamples_test[cls]

    LocalDist_train = [[0.0 for _ in range(10)] for __ in range(NUM_OF_CLIENTS)]
    for client in range(NUM_OF_CLIENTS):
        for cls in CategoryToClients[client]:
            count[cls] -= 1
            if count[cls] != 0:
                LocalDist_train[client][cls] = int(RemainedSamples_train[cls] / Categories[cls])*1.0
            else:
                LocalDist_train[client][cls] = 1.0 * (RemainedSamples_train[cls] - (Categories[cls] - 1) * int(RemainedSamples_train[cls] / Categories[cls]))

    LocalDist_test_array=np.array(LocalDist_test)
    LocalDist_train_array=np.array(LocalDist_train)
    np.savetxt("./data/LocalDist_niid_test5_4.txt", LocalDist_test_array, fmt='%d')
    np.savetxt("./data/LocalDist_niid5_4.txt", LocalDist_train_array, fmt='%d')
    '''
    LocalDist_array=np.loadtxt("./data/LocalDist_niid4.txt")
    LocalDist_test_array=np.loadtxt("./data/LocalDist_niid_test4.txt")

    SamplesToClients=[0.0 for i in range(NUM_OF_CLIENTS)]
    for client in range(NUM_OF_CLIENTS):
        SamplesToClients[client] = sum(LocalDist_array[client, :])

    SamplesToClients_test=[0.0 for i in range(NUM_OF_CLIENTS)]
    for client in range(NUM_OF_CLIENTS):
        SamplesToClients_test[client] = sum(LocalDist_test_array[client, :])

    LocalDist=LocalDist_array.tolist()
    LocalDist_test=LocalDist_test_array.tolist()

    MAX_NUM = max(SamplesToClients)
    return CategoryToClients, Categories_client, LocalDist, LocalDist_test, SamplesToClients, SamplesToClients_test, MAX_NUM