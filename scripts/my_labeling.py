__authors__ = '1531236, 1532874, 1526000'
__group__ = 'DL.15'

import numpy as np
from Kmeans import *
from KNN import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageOps
from time import time

#FUNCIONS D'ANALISI QUALITATIU
def retrievalByColor(imatges, resKmeans, llistaC, isok = None):
    answ = []
    for i,el in enumerate(resKmeans):
        count = 0
        for color in llistaC:
            if color in el:
                count += 1
        if count == len(llistaC):
            answ.append(imatges[i])
            if isok is not None:
                sum = 0
                for revisa in llistaC:
                    if revisa in test_color_labels[i]:
                        sum += 1
                if sum == len(llistaC):
                    isok.append(True)
                else:
                    isok.append(False)

    return answ


def Retrival_by_shape(llimatges, etiquetes, cerca, isok=None):
    llista = []
    for i,x in enumerate(etiquetes):
        if x == cerca:
            llista.append(llimatges[i])
            if isok is not None:
                if x == test_class_labels[i]:
                    isok.append(True)
                else:
                    isok.append(False)

    return llista


def retrieval_combined(imatges, formes, colors, forma, color):
    answ = []
    for i, el in enumerate(imatges):
        if forma == formes[i] and color in colors[i]:
            answ.append(el)
    return answ



#FUNCIONS D'ANALISI QUANTITATIU
def kmean_statistics(class_Kmeans, kmax):
    # cal mostrar WCD, nombre d'iteracions que ha necessitat per convergir, etc.
    k = 2
    print("gerard borra aquest missatge")
    while (k <= kmax):
        class_Kmeans.K = k
        print("------------ Attempt: k =", k, "----------------")

        it, time_converges = class_Kmeans.fit()
        wcd = class_Kmeans.whitinClassDistance('intraclass')

        print("Iterations: ", it)
        print("Time until converges (s): ", time_converges)
        print("WCD: ", wcd)
        k += 1
    class_Kmeans.find_bestK(kmax, 'intraclass')
    print("Best K: ", class_Kmeans.K)


def get_shape_accuracy(resKNN, labels):
    si = 0
    for i, el in enumerate(resKNN):
        if el == labels[i]:
            si += 1
    return (si/len(resKNN))*100


def get_color_accuracy(resKmeans, labels):
    encert = 0
    for i, el in enumerate(labels):
        it = 0
        for color in el:
            if color in resKmeans[i]:
                it += 1
        if it == len(el):
            encert += 1
    return (encert/len(resKmeans))*100



if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='../images/', gt_json='../images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    #Kmeans
    resKmeans = []
    print("go")
    time_until_converges = []
    for el in test_imgs[0:50]:
        starting_time = time()
        answer = KMeans(el)
        answer.options['km_init'] = 'random'
        answer.find_bestK(10, 'fisher')
        answer.fit()
        end_time = time()
        timet = end_time - starting_time
        time_until_converges.append(timet)

        #Plot3DCloud(answer)
        #visualize_k_means(answer, [80, 60, 3])
        resKmeans.append(get_colors(answer.centroids))
        #print(answer.centroids)
        #print(get_colors(answer.centroids))
    temps = np.median(np.array(time_until_converges))
    print(temps)

    #RETRIEVAL_BY_COLOR
    isok = []
    retrievedc = retrievalByColor(test_imgs[0:50], resKmeans, ["Black"], isok)
    if len(isok) != 0:
        #GET_COLOR_ACCURACY
        percent = get_color_accuracy(resKmeans, test_color_labels[0:50])
        print("We color labbeled a", percent, "% of the images")
    answ = []

    if len(retrievedc) == 0:
        print("Cap imatge trobada")

    else:
        for i,el in enumerate(retrievedc):
            im = Image.fromarray(el)
            if isok[i]:
                imagenconborde = ImageOps.expand(im, border=5, fill="green")
            else:
                imagenconborde = ImageOps.expand(im, border=5, fill="red")
            answ.append(imagenconborde)

        visualize_retrieval(answ, len(answ))


    #RETRIEVAL BY SHAPE
    #passem les imatges en b/n
    answ = []
    for el in train_imgs:
        answ.append(cv2.cvtColor(el, cv2.COLOR_BGR2GRAY))
    a = np.array(answ)

    answtest = []
    for el in test_imgs:
        answtest.append(cv2.cvtColor(el, cv2.COLOR_BGR2GRAY))
    b = np.array(answtest)

    #Entrenem l'algorisme
    resultatKNN = []
    knntest = KNN(a, train_class_labels)

    #afegim les imatges sobre les que volem buscar
    hola = knntest.predict(b[0:50], 8)
    #realitzem la busqueda sobre les etiquetes obtingudes
    isok = []
    retrievalbyshape = Retrival_by_shape(test_imgs[0:50], hola, "Dresses", isok)
    if len(retrievalbyshape) == 0:
        print("No he trobat res, et puc buscar", classes)
    else:
        answ = []
        for i, el in enumerate(retrievalbyshape):
            im = Image.fromarray(el)
            if isok[i]:
                imagenconborde = ImageOps.expand(im, border=5, fill="green")
            else:
                imagenconborde = ImageOps.expand(im, border=5, fill="red")
            answ.append(imagenconborde)

        visualize_retrieval(answ, len(answ))


    #GET_SHAPE_ACCURACY
    perc = get_shape_accuracy(hola, test_class_labels[0:50])

    print("Hem encertat un ", perc, "% en l'etiquetatge de forma")


    #RETRIEVAL COMBINED
    si = retrieval_combined(test_imgs[0:50], hola, resKmeans, "Shorts", "Black")
    if len(si) != 0:
        visualize_retrieval(si, len(si))
    else:
        print("No he trobat res")


    #KMEAN_STATISTICS
    #kmean_statistics(answer, 20)








