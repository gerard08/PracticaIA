__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
from Kmeans import *
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def retrievalByColor(imatges, resKmeans, llistaC):
    answ = []
    for i,el in enumerate(resKmeans):
        count = 0
        for color in llistaC:
            if color in el:
                count += 1
        if count == len(llistaC):
            answ.append(imatges[i])

    return answ



if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='../images/', gt_json='../images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    #Kmeans
    resKmeans = []
    for el in test_imgs[0:20]:
        answer = KMeans(el)
        answer.options['km_init'] = 'random'
        answer.find_bestK(10)
        answer.fit()
        resKmeans.append(get_colors(answer.centroids))

    #retrieve by color
    retrievedc = retrievalByColor(test_imgs[0:20], resKmeans, ["Blue"])

    answ = []

    if len(retrievedc) == 0:
        print("cap imatge trobada")

    else:
        for el in retrievedc:
            answ.append(Image.fromarray(el))

        twidth = 0
        for el in answ:
            widths, heights = el.size
            twidth += widths
        total_width = twidth
        max_height = heights

        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in answ:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.show()

## You can start coding your functions here

#bestk
#retrieval by color








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




