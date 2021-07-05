from __future__ import print_function
from __future__ import division

import os, glob
import pathlib
import math
import operator
import random
import logging

if __name__ == "__main__":
    #carpeta = '/Users/jjrau/OneDrive/Desktop/ETSINF/TFG - CLASIFICACIÓN DE TEXTOS MANUSCRITOS/code/tfidf_test_train/train'
    carpeta = '/Users/jflov/OneDrive/Escritorio/TFG/tfidf_test_train/train'
    directorio  = os.listdir(carpeta)
    cont = 0
    dic={} #key -> (número páginas,clase), valor -> número exp con esas páginas
    dic_class = {} #key -> clase, valor -> número de páginas
    print('Train')
    print('*'*50)
    for doc in directorio:
        #pages_56-59_o.idx
        c = doc.split('.')[0].split('_')[-1]
        n1 = doc.split('.')[0].split('_')[1].split('-')[0]
        n2 = doc.split('.')[0].split('_')[1].split('-')[-1]
        res = (int(n2) - int(n1)) + 1
        if (res,c) not in dic:
            dic[(res,c)] = 1
        else:
            dic[(res,c)] += 1

        if c not in dic_class:
            dic_class[c] = res
        else:
            dic_class[c] += res
    #carpeta = '/Users/jjrau/OneDrive/Desktop/ETSINF/TFG - CLASIFICACIÓN DE TEXTOS MANUSCRITOS/code/tfidf_test_train/test'
    carpeta = '/Users/jflov/OneDrive/Escritorio/TFG/tfidf_test_train/test'
    directorio  = os.listdir(carpeta)
    cont = 0
    print('Test')
    print('*'*50)
    for doc in directorio:
        #pages_56-59_o.idx
        c = doc.split('.')[0].split('_')[-1]
        n1 = doc.split('.')[0].split('_')[1].split('-')[0]
        n2 = doc.split('.')[0].split('_')[1].split('-')[-1]
        res = (int(n2) - int(n1)) + 1
        
        if (res,c) not in dic:
            dic[(res,c)] = 1
        else:
            dic[(res,c)] += 1

        dic_class[c] += res
    # print('Numero de Paginas',"  ", "Numero de expedientes con esas paginas")
    # print('*'*50)

    # print('Clase'," ","Numero de Paginas")
    # print('*'*50)
    # for key in sorted(dic_class.keys()) :
    #     print(key , " :: " , dic_class[key])
    
    #print(dic)
    print('*'*50)
    for n,c in dic:
        if int(n) <= 5:
            print('Clase:',c,'Numero de paginas',dic[n,c])
