__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
from Kmeans import *
from KNN import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageOps

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

def kmean_statistics(class_Kmeans, kmax):
    # cal mostrar WCD, nombre d'iteracions que ha necessitat per convergir, etc.
    k=2
    while(k<=kmax):
        class_Kmeans.k = k
        iterations, time_until_converges = class_Kmeans.fit()
        wcd = class_Kmeans.whitinClassDistance()
        print("------------ Attempt: k =",k,"----------------")
        print("Iterations: ",iterations)
        print("Time until converges (s): ",time_until_converges)
        print("WCD: ",wcd)
        k = k + 1

def Retrival_by_shape(llimatges, etiquetes, cerca):
    llista = []
    for i,x in enumerate(etiquetes):
        if x == cerca:
            llista.append(llimatges[i])

    return llista

def retrieval_combined(imatges, formes, colors, forma, color):
    answ = []
    for i, el in enumerate(imatges):
        if forma == formes[i] and color in colors[i]:
            answ.append(el)
    return answ

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

def get_shape_accuracy(resKNN, labels):
    si = 0
    for i, el in enumerate(labels):
        if el == labels[i]:
            si += 1
    return (si/len(resKNN))*100


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='../images/', gt_json='../images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    #Kmeans
    resKmeans = []
    for el in test_imgs[4:5]:
        answer = KMeans(el)
        answer.options['km_init'] = 'random'
        answer.find_bestK(8,'fisher')
        answer.fit(True)
        #Plot3DCloud(answer)
        #visualize_k_means(answer, [80, 60, 3])
        resKmeans.append(get_colors(answer.centroids))
        #print(answer.centroids)
        #print(get_colors(answer.centroids))


    #retrieve by color
    isok = []
    retrievedc = retrievalByColor(test_imgs[4:5], resKmeans, ["Black"], isok)
    if len(isok) != 0:
        percent = get_color_accuracy(resKmeans, test_color_labels[4:5])
        print("hem encertat un ", percent, "% en l'etiquetatge de color")
    answ = []

    if len(retrievedc) == 0:
        print("cap imatge trobada")

    else:
        for i,el in enumerate(retrievedc):
            im = Image.fromarray(el)
            if isok[i]:
                imagenconborde = ImageOps.expand(im, border=5, fill="green")
            else:
                imagenconborde = ImageOps.expand(im, border=5, fill="red")
            answ.append(imagenconborde)

        visualize_retrieval(answ, len(answ))

    #stadistics


## You can start coding your functions here

#kmeans_statistics
#kmean_statistics(answer, 10)
''''
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
retrievalbyshape = Retrival_by_shape(test_imgs[0:50], hola, "Shorts")
if len(retrievalbyshape) == 0:
    print("no he trobat res :(, et puc buscar", classes)
else:
    visualize_retrieval(retrievalbyshape, len(retrievalbyshape))

#busqueda conjunta

si = retrieval_combined(test_imgs[0:50], hola, resKmeans, "Shorts", "Brown")
if len(si) != 0:
    visualize_retrieval(si, len(si))
else:
    print("no he trobat res")

perc = get_shape_accuracy(hola, test_class_labels[0:50])
print(perc, "percent d'accuracy en la detecció de forma")
#       RESUM DEL VIDEO

#visualize_Kmeans(Kmeans, [80,60,3](tamany imatge))

#visualize_retrieval(Imatges_ordenades, info(info que volguem mostrar), ok(array amb true i false), title='Query: Socks')

#Retrieval_by_color(Imatges, Resultat_Kmeans, "pink")-->Retorna imatges del color que li demanem

#Get_color_accuracy(Resultat Kmeans, Ground_Truth) --> Ens dona un resultat expressat amb un numero
#ell fa exemple amb la K per exemple
#les imatges tenen diferents colors, aixi que hem de "jugar" amb l'acuracy per trobar la ideal

#Get_shape_accuracy(suposo que es el mateix que el color acuracy)

#Kmeans_statistics:
#nº iteracions per convergir (+ eficients = - iteracions) o tambe podriem posar el temps
#WithinClassDistance de cada K(compararles)

#millores

    #KNN
#Cambiar tamany del Train set
#canviar l'espai de característiques
    #Posar les imatges a color (ara mateix estan en B/N)
    #Diferents resolucions d'imatge
    #Altres característiques
        #Caracteristica 1: total de pixels blanc
        #2: dif entre pixels imatge sup i imatge inf (p. ex pantalons son rectes pero un bolso no)
        #3: variança valors dels pixels

    #Kmeans
#inicialitzar centroides d'una altra manera
#treballar amb diferents espais de color (Lab, Hsv, Colornaming,...)
#Utilitzar altres heurístiques per trobar la bestK(InterClassDistance, Fisher...)
#Podriem jugar amb el llindar de colors, nosaltres ara fem servir un 20%, pero el podriem canviar

'''


