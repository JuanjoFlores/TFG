from __future__ import print_function
from __future__ import division

import os, glob
import pathlib
import math
import operator
import random
import logging

if __name__ == "__main__":

    carpeta = '/Users/jjrau/OneDrive/Desktop/ETSINF/TFG - CLASIFICACIÓN DE TEXTOS MANUSCRITOS/code/tfidf_test_train/train'
    #carpeta = '/Users/jflov/OneDrive/Escritorio/TFG - CLASIFICACIÓN DE TEXTOS MANUSCRITOS/code/tfidf_test_train/train'
    directorio  = os.listdir(carpeta)
    m = [] #TODOS LOS DOCUMENTOS
    fD = {} #Diccionario, key -> Doc, valor -> Esperanza total de palabras.
    f_vD = {} #Diccionario key -> (doc,word) valor -> estimación del numero de veces que aparece una palabra v en un doc.
    tf = {} #Diccionario key -> (Doc, palabra), valor-> Tf
    f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> palabra, valor -> num_docs
    for doc in directorio:
        ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
        #print("Loading {}".format(path))
        path = carpeta + '/' + doc
        f = open(path, "r")
        lines = f.readlines() 
        f.close()
        acum = 0 #Acumulador de las probabilidades de las palabras del doc
        lw = [] #set con todas las palabras     
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            word = line.split()[0] #Split por espacios en blanco
            prob_word = line.split()[1:] #Cogemos la probabilidad
            if(float(prob_word[0]) > 0.1):
                #Calculo de fD -> Esperanza del numero total de palabras en X
                acum += float(prob_word[0])
                r = f_vD.get((doc, word), 0)
                r += float(prob_word[0])
                f_vD[doc,word] = r
                lw.append(word)
                #Calculo de f(tv)
                s = f_tv.get(word,set()) #Devuelve el valor de word que es un set de docs si ya esta word y si no un set vacío.
                s.add(doc)
                f_tv[word] = s #f(tv) final es len(f_tv[word])
        fD[doc] = acum
        lw = set(lw)
        #Tf= f(v,D)/f(D)
        for word in lw:
            if (doc, word) in f_vD:
                tf[doc, word] = f_vD[doc, word]/fD[doc]
        m.append(doc)
       
        
    # #Calculo de Idf:
    idf = {} #Diccionario key -> palabra, valor -> idf
    for word in f_tv:
            idf[word] = math.log2(len(m)/len(f_tv[word]))
    

    # #Calculo de Tf*Idf:
    tf_Idf = {} #Diccionario key -> tupla(doc, palabra), valor -> tf*idf
    for doc in m:
        for word in f_tv:
            if (doc,word) in tf:
                tf_Idf[doc,word] = tf[doc,word]*idf[word]
            else:
                tf_Idf[doc,word] = 0.0

    words = []
    with open('resultadosIG.txt', 'r') as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            word = line.split()[0] #Split por espacios en blanco
            words.append(word)
    
            
    with open('vector_tf_idf_m_train.txt', 'w') as f:
        for word in words:
            s = word + ' '
            f.write(s)
        s = '\n'
        f.write(s)
        for doc in m:
            d = doc + ' '
            f.write(d)
            for pal in words:
                n = str(tf_Idf[doc,pal]) + ' '
                f.write(n)
            s = '\n'
            f.write(s)
    