__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #mirem si el tipus de dades es float, en cas que no ho sigui ho convertim
        if train_data.dtype is not float:
            train_data.astype(float)

        #si la matriu no te dues dimensions la convertim a una de dues dimensions
        if len(train_data.shape) is not 2:
            #basicament 0 seria l'altura, 1 la llargada i 2 la profunditat en una matriu de 3d
            #llavors per passarla a una de 2d, ens quedem amb l'altura original i multipliquem llargada
            #i profunditat per saber el numero de pixels de la imatge
            train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])

        self.train_data = train_data


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        if len(test_data.shape) is not 2:
            test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        dist = cdist(test_data, self.train_data)

        self.neighbors = self.labels[np.argmin(dist, axis=1)]




        print()





    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        self.get_class()
        print()



        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)
