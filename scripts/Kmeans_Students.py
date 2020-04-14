__authors__ = ['XXXXXXXXX','YYYYYYYY']
__group__ = 'GrupZZ'

import numpy as np
import utils
import math
import copy
from scipy.spatial import distance as d


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options


    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """

        #Si les dades de X no son de tipus float, les convertim a float
        if X.dtype != 'float64':
            X = X.astype('float64')

        #basicament ens demanen que tornem la matriu que ens passen com una matriu de dues dimensions
        #on les files siguin els pixels i les columnes el RGB, o com a minim així ho he entès jo

        if len(X.shape) != 2:
            #calculem el numero de pixels que hi ha a la matriu fent Fila x Columna
            npixels = X.shape[0] * X.shape[1]

            #ho convertim en una matriu de dues dimensions mitjançant l'eina "reshape"
            self.X = X.reshape(npixels, X.shape[len(X.shape)-1])


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def afegir(self, desti, element):   #funció feta per mi per comprovar la repetició d'elements dins la matriu
        p = 0
        afegir = True

        while p < self.K and afegir == True:
            if (desti[p] == element).all():
                afegir = False
            p += 1
        return afegir


    def _init_centroids(self):
        """
        Initialization of centroids
        """
        # creo una matriu amb la mida del output pero buida (plena de 0s)

        afegits = i = 0
        punts = np.zeros(shape=(self.K, self.X.shape[1]))

        if self.options['km_init'].lower() == 'first':
            #mentres no haguem afegit tants elements com per completar totes les files de la matriu
            while afegits != self.K:

                if self.afegir(punts, self.X[i]):
                    #igualem la fila amb el pixel a afegir
                    punts[afegits] = self.X[i]
                    afegits += 1
                i += 1


        elif self.options['km_init'].lower() == 'random':

            while afegits != self.K:

                auxr = np.random.randint(low=0, high=self.X.shape[0], size=(1))[0]

                if self.afegir(punts, self.X[auxr]):
                    #igualem la fila amb el pixel a afegir
                    punts[afegits] = self.X[auxr]
                    afegits += 1
                i += 1


        else:   #pendent de fer

            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids =np.random.rand(self.K, self.X.shape[1])

        self.centroids = punts


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################


        if self.centroids is not None : self.old_centroids = copy.deepcopy(self.centroids)

        #calculo els nous centroids

        centroids = {}

        for n, cent in enumerate(self.labels):
            values = [self.X[n][0], self.X[n][1], self.X[n][2], 1]
            #vaig sumant els valors de RGB
            if cent in centroids.keys():
                centroids[cent][0] += values[0]
                centroids[cent][1] += values[1]
                centroids[cent][2] += values[2]
                #nº elements sumats (per fer despres mitjana)
                centroids[cent][3] += 1
            else:
                #si no hi es l'afegeixo
                centroids[cent] = values
        #faig mitjana
        for i, cent in enumerate(self.centroids):
            cent[0] = centroids[i][0]/centroids[i][3]
            cent[1] = centroids[i][1] / centroids[i][3]
            cent[2] = centroids[i][2] / centroids[i][3]


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################


        # Crec que així ja esta ben fet, caldria revisar
        if np.allclose(self.old_centroids, self.centroids, self.options['tolerance']):
            return True
        else:
            return False


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        difference = False
        iter = 0

        #Comprova si convergeix i si el num d'iteracions es menor al permes
        while difference == False and iter <= self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            difference = self.converges()
            iter += 1
        self.num_iter = iter

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        dist = distance(self.X, self.centroids)

        #trec la distancia de cada pixel amb el centroide mes proper
        total_dist = dist.min(1)

        #faig el calcul de intra-class per cada x i faig el total
        total = np.sum(total_dist**2)

        #calcul de la mitjana
        wcd = total / total_dist.shape[0]

        return wcd

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        cadak = 0
        wcd = np.zeros(max_K - 1)
        dec = np.zeros(max_K - 2)

        #calcula cada intra-class de cadascuna de les k
        while cadak <= max_K - 2:
            self.K = cadak + 2
            self.fit()
            wcd[cadak] = self.whitinClassDistance()
            cadak += 1
        print()

        k = 1
        #ara es calcula el llindar de la diferencia entre les diferents k
        while k+1 < max_K:
            dec[k-1] = 100 - (100 * (wcd[k] / wcd[k-1]))
            k = k + 1
        print()

        #mejor K
        self.K = dec.argmax() + 3
        print()


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    '''
    # creo una matriu buida de tamany PxK
    dist = np.zeros((X.shape[0], C.shape[0]))
    i = 0
    # from scipy.spatial import distance
    for num, centroid in enumerate(C):
        for pixel in X:
            dist[i][num] = math.sqrt(
                pow((pixel[0] - centroid[0]), 2) + pow((pixel[1] - centroid[1]), 2) + pow((pixel[2] - centroid[2]), 2))
            # dist[i][num] = distance.euclidean(pixel, centroid)
            if i < X.shape[0] - 1:
                i += 1
            else:
                i = 0

    #creo una matriu buida de tamany PxK
    '''
    dist = np.zeros((X.shape[0], C.shape[0]))

    for num, centroid in enumerate(C):
        for i, pixel in enumerate(X):
            aux = math.sqrt(pow((pixel[0] - centroid[0]), 2) + pow((pixel[1] - centroid[1]), 2) + pow((pixel[2] - centroid[2]), 2))
            dist[i][num] = aux
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
