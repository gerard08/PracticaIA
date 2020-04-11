__authors__ = ['XXXXXXXXX','YYYYYYYY']
__group__ = 'GrupZZ'

import numpy as np
import utils
import math


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################






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
        punts = np.zeros(shape=(self.K, self.X.shape[1]))

        afegits = i = 0

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

                if self.afegir(punts, np.random.rand(self.K, self.X.shape[1])):
                    # igualem la fila amb el pixel a afegir
                    punts[afegits] = np.random.rand(self.K, self.X.shape[1])
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
        self.get_labels = np.argmin(dist, axis=1)


        #self.labels = np.random.randint(self.K, size=self.X.shape[0])

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.rand()

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    #creo una matriu buida de tamany PxK
    dist = np.zeros((X.shape[0], C.shape[0]))
    i = 0
    #from scipy.spatial import distance
    for num, centroid in enumerate(C):
        for pixel in X:
            dist[i][num] = math.sqrt(pow((pixel[0] - centroid[0]), 2) + pow((pixel[1] - centroid[1]), 2) + pow((pixel[2] - centroid[2]), 2))
            #dist[i][num] = distance.euclidean(pixel, centroid)
            if i < X.shape[0] - 1:
                i += 1
            else:
                i = 0

    return dist


    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    #return np.random.rand(X.shape[0], C.shape[0])


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
