'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import logging




class Dataset(object):
    '''

    '''

    def __init__(self, ratio):
        self.ratio = ratio
        self.trainMatrix, self.validationMatrix, self.testMatrix = self.load_data()
        self.num_users, self.num_items = self.trainMatrix.shape
    def load_data(self):
        filename = '/DATA2/data/kncui/data/ml-1m/data.txt'
        # Get number of users and items
        num_users, num_items, interactions = 0, 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            interactions = interactions + 1
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
                interactions = interactions + 1
        # Construct matrix
        test_mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        vali_mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        train_mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        interaction = 0
        with open(filename, "r") as f:
            line = f.readline()
            interaction = interaction + 1
            while line != None and line != "":
                arr = line.split(" ")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if interaction <= interactions * self.ratio:
                    train_mat[user, item] = rating
                elif interaction <= (interactions * self.ratio + 0.5*(1 - self.ratio)):
                    vali_mat[user, item] = rating
                else:
                    test_mat[user, item] = rating
                line = f.readline()
                interaction = interaction + 1
        return train_mat, vali_mat, test_mat


def get_train_instances(train):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(train[u, i])
    return user_input, item_input, labels


def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)