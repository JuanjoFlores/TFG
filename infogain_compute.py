from __future__ import print_function
from __future__ import division

import os, glob
import pathlib
import math
import operator
import random
import logging


if __name__ == "__main__":
    ###CARGAMOS PAGINAS ARCHIVO .idx
    print("CARGANDO TODOS LOS ARCHIVOS Y SUS PALABRAS")
    #carpeta = '/home/jflores/paquete/tfidf_test_train/train'
    carpeta = '/Users/jjrau/OneDrive/Desktop/ETSINF/TFG - CLASIFICACIÓN DE TEXTOS MANUSCRITOS/code/tfidf_test_train/train'
    directorio  = os.listdir(carpeta)
    m = [] #TODOS LOS DOCUMENTOS
    #DOCUMENTOS DE CADA CLASE
    m_cen = []
    m_o = []
    m_t = []
    m_cp = []
    m_r = []
    m_a = []
    m_v = []
    m_p = []
    m_s = []
    f_tv = {} #f(tv) -> diccionario con la key palabra, valor set de los documentos donde esta la palabra.
    #Diccionario para cada clase donde: key -> palabra, valor ->num_docs de la clase donde aparece
    #f(c, tv) (Numero de documentos de la clase c en los que esta la palabra v) Diccionario por clase donde -> Key:palabra, valor-> n_docs
    f_cen_tv = {}
    f_o_tv = {}
    f_t_tv={}
    f_cp_tv = {}
    f_r_tv={}
    f_a_tv={}
    f_v_tv={}
    f_p_tv={}
    f_s_tv={} 
    for doc in directorio:
        ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
        #pages_56-59_o.idx
        n_clase = doc.split('.')[0].split('_')[-1] #Contiene nombre de clase
        if n_clase == "cen":
            m_cen.append(doc)
        elif n_clase == "o":
            m_o.append(doc)
        elif n_clase == "t":
            m_t.append(doc)
        elif n_clase == "cp":
            m_cp.append(doc)
        elif n_clase == "r":
            m_r.append(doc)
        elif n_clase == 'a':
            m_a.append(doc)
        elif n_clase == 'v':
            m_v.append(doc)
        elif n_clase == 'p':
            m_p.append(doc)
        elif n_clase == 's':
            m_s.append(doc)
        #print("Loading {}".format(path))
        path = carpeta + '/' + doc
        f = open(path, "r")
        lines = f.readlines() 
        f.close()
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            word = line.split()[0] #Split por espacios en blanco
            prob_word = line.split()[1:] #Cogemos la probabilidad
            if(float(prob_word[0]) > 0.1):
                #Calculo de f(tv)
                r = f_tv.get(word,set()) #Devuelve el valor de word que es un set de docs si ya esta word y si no un set vacío.
                r.add(doc)
                f_tv[word] = r #f(tv) final es len(f_tv[word])
                #Calculo de f(c,tv)
                if n_clase == "cen":
                    r = f_cen_tv.get(word,set())
                    r.add(doc)
                    f_cen_tv[word] = r
                if n_clase == "o":
                    r = f_o_tv.get(word,set())
                    r.add(doc)
                    f_o_tv[word] = r
                if n_clase == "t":
                    r = f_t_tv.get(word,set())
                    r.add(doc)
                    f_t_tv[word] = r
                if n_clase == "cp":
                    r = f_cp_tv.get(word,set())
                    r.add(doc)
                    f_cp_tv[word] = r
                if n_clase == "r":
                    r = f_r_tv.get(word,set())
                    r.add(doc)
                    f_r_tv[word] = r
                if n_clase == "a":
                    r = f_a_tv.get(word,set())
                    r.add(doc)
                    f_a_tv[word] = r
                if n_clase == "v":
                    r = f_v_tv.get(word,set())
                    r.add(doc)
                    f_v_tv[word] = r
                if n_clase == "p":
                    r = f_p_tv.get(word,set())
                    r.add(doc)
                    f_p_tv[word] = r
                if n_clase == "s":
                    r = f_s_tv.get(word,set())
                    r.add(doc)
                    f_s_tv[word] = r

        m.append(doc)
    
    ###CALCULO DE IG (INFORMATION GAIN) PARA CADA PALABRA
    
   
    print("PASANDO A CALCULAR LAS ECUACIONES 2")
    #ECUACIONES 2:
    ##Calculo de p(tv) = f(tv)/M y p(no_tv) = M - f(tv)/M -> Probabilidad de que algun doc contenga la palabra v
    p_tv = {} #Diccionario key -> palabra, valor -> probabilidad que algun doc contenga la palabra v
    p_notv = {}
    for word in f_tv:
        p_tv[word] = len(f_tv[word])/len(m)
        p_notv[word] = (len(m) - len(f_tv[word])) / len(m)
    print("PASANDO A CALCULAR LAS ECUACIONES 3")
    #ECUACIONES 3:
    ##Calculo de P(c|tv) = f(c,tv)/f(tv) y P(c|no_tv) = Mc - f(c,tv) / M - f(tv)  -> Probabilidad condicional de que un documento pertenzca a la clase c,
    #dado que contiene la palabra v
    #Dicionarios donde key -> palabra, valor-> probabilidad de que un doc en el que esta esa palabra pertenezca a esa clase
    p_cen_tv = {}
    p_o_tv = {}
    p_t_tv = {}
    p_cp_tv = {}
    p_r_tv = {}
    p_a_tv = {}
    p_v_tv = {}
    p_p_tv = {}
    p_s_tv = {}
    
    #Negados
    p_cen_notv = {}
    p_o_notv = {}
    p_t_notv = {}
    p_cp_notv = {}
    p_r_notv = {}
    p_a_notv = {}
    p_v_notv = {}
    p_p_notv = {}
    p_s_notv = {}

    
    #Para cada palabra calculamos:
    for word in f_tv:
        
        part = (len(m) - len(f_tv[word]))
        #CLASE CEN:
        if word in f_cen_tv:
            p_cen_tv[word] = len(f_cen_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_cen_notv[word] = (len(m_cen) - len(f_cen_tv[word])) / part
            else:
                p_cen_notv[word] = 0

        #CLASE O:
        if word in f_o_tv:
            p_o_tv[word] = len(f_o_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_o_notv[word] = (len(m_o) - len(f_o_tv[word])) / part
            else:
                p_o_notv[word] = 0 

        #CLASE T:
        if word in f_t_tv:
            p_t_tv[word] = len(f_t_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_t_notv[word] = (len(m_t) - len(f_t_tv[word])) / part
            else:
                p_t_notv[word] = 0
    
        #CLASE CP:
        if word in f_cp_tv:
            p_cp_tv[word] = len(f_cp_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_cp_notv[word] = (len(m_cp) - len(f_cp_tv[word])) / part
            else:
                p_cp_notv[word] = 0

        #CLASE R:
        if word in f_r_tv:    
            p_r_tv[word] = len(f_r_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_r_notv[word] = (len(m_r) - len(f_r_tv[word])) / part
            else:
                p_r_notv[word] = 0

        #CLASE A:
        if word in f_a_tv:
            p_a_tv[word] = len(f_a_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_a_notv[word] = (len(m_a) - len(f_a_tv[word])) / part
            else:
                p_a_notv[word] = 0

        #CLASE V:
        if word in f_v_tv:
            p_v_tv[word] = len(f_v_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_v_notv[word] = (len(m_v) - len(f_v_tv[word])) / part
            else:
                p_v_notv[word] = 0
        
        #CLASE P:
        if word in f_p_tv:
            p_p_tv[word] = len(f_p_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_p_notv[word] = (len(m_p) - len(f_p_tv[word])) / part
            else:
                p_p_notv[word] = 0
        
        #CLASE S:
        if word in f_s_tv:
            p_s_tv[word] = len(f_s_tv[word])/len(f_tv[word])
            #Parte negada
            if(part > 0):
                p_s_notv[word] = (len(m_s) - len(f_s_tv[word])) / part
            else:
                p_s_notv[word] = 0
    
    
    print("PASANDO A CALCULAR LA ECUACIÓN FINAL DEL IG")
    #ECUACIÓN 1:
    ##Calculo del information gain para cada palabra
    infGain = {} #Diccionario donde: key -> palabra, valor -> information gain
    n_clases = 9
    p_c = 1/n_clases #Probabilidad a priori para cada clase (9 clases y todas tienen la misma probabilidad)
    res = (-(p_c * math.log2(p_c)))*n_clases #Primera parte de la ecuación
    for word in f_tv:
        #Segunda parte de la ecuación
        if word in p_cen_tv:
            if p_cen_tv[word] > 0:
                log_cen = math.log2(p_cen_tv[word])
            else:
                log_cen = 0
            p_cen = p_cen_tv[word]
        else:
             p_cen = 0
             log_cen = 0

        if word in p_o_tv:
            if p_o_tv[word] > 0:
                log_o = math.log2(p_o_tv[word])
            else:
                log_o = 0
            p_o = p_o_tv[word]
        else:
            p_o = 0
            log_o = 0
        
        if word in p_t_tv:
            if p_t_tv[word] > 0:
                log_t = math.log2(p_t_tv[word])
            else:
                log_t = 0
            p_t = p_t_tv[word]
        else:
            p_t = 0
            log_t = 0
        
        if word in p_cp_tv:
            if p_cp_tv[word] > 0:
                log_cp = math.log2(p_cp_tv[word])
            else:
                log_cp = 0
            p_cp = p_cp_tv[word]
        else:
            p_cp = 0
            log_cp = 0

        if word in p_r_tv:
            if p_r_tv[word] > 0:
                log_r = math.log2(p_r_tv[word])
            else:
                log_r = 0
            p_r = p_r_tv[word]
        else:
            p_r = 0
            log_r = 0

        if word in p_a_tv:
            if p_a_tv[word] > 0:
                log_a = math.log2(p_a_tv[word])
            else:
                log_a = 0
            p_a = p_a_tv[word]
        else:
            p_a = 0
            log_a = 0

        if word in p_v_tv:
            if p_v_tv[word] > 0:
                log_v = math.log2(p_v_tv[word])
            else: 
                log_v = 0
            p_v = p_v_tv[word]
        else:
            p_v = 0
            log_v = 0

        if word in p_p_tv:
            if p_p_tv[word] > 0:
                log_p = math.log2(p_p_tv[word])
            else: 
                log_p = 0
            p_p = p_p_tv[word]
        else:
            p_p = 0
            log_p = 0

        if word in p_s_tv:
            if p_s_tv[word] > 0:
                log_s = math.log2(p_s_tv[word])
            else: 
                log_s = 0
            p_s = p_s_tv[word]
        else:
            p_s = 0
            log_s = 0

        res2 = p_tv[word]*(p_cen*log_cen + 
            p_o*log_o + p_t*log_t +
            p_cp*log_cp + p_r* log_r+
            p_a*log_a + p_v* log_v+
            p_p*log_p + p_s*log_s)
        
        #Tercera parte de la ecuación

        if word in p_cen_notv:
            if p_cen_notv[word] > 0:
                log_cen_n = math.log2(p_cen_notv[word])
            else:
                log_cen_n = 0
            p_cen_n = p_cen_notv[word]
        else:
             p_cen_n = 0
             log_cen_n = 0

        if word in p_o_notv:
            if p_o_notv[word] > 0:
                log_o_n = math.log2(p_o_notv[word])
            else:
                log_o_n = 0
                p_o_n = p_o_notv[word]
        else:
            p_o_n = 0
            log_o_n = 0
        
        if word in p_t_notv:
            if p_t_notv[word] > 0:
                log_t_n = math.log2(p_t_notv[word])
            else:
                log_t_n = 0
                p_t_n = p_t_notv[word]
        else:
            p_t_n = 0
            log_t_n = 0
        
        if word in p_cp_notv:
            if p_cp_notv[word] > 0:
                log_cp_n = math.log2(p_cp_notv[word])
            else:
                log_cp_n = 0
                p_cp_n = p_cp_notv[word]
        else:
            p_cp_n = 0
            log_cp_n = 0

        if word in p_r_notv:
            if p_r_notv[word] > 0:
                log_r_n = math.log2(p_r_notv[word])
            else:
                log_r_n = 0
                p_r_n = p_r_notv[word]
        else:
            p_r_n = 0
            log_r_n = 0

        if word in p_a_notv:
            if p_a_notv[word] > 0:
                log_a_n = math.log2(p_a_notv[word])
            else:
                log_a_n = 0
                p_a_n = p_a_notv[word]
        else:
            p_a_n = 0
            log_a_n = 0

        if word in p_v_notv:
            if p_v_notv[word] > 0:
                log_v_n = math.log2(p_v_notv[word])
            else: 
                log_v_n = 0
                p_v_n = p_v_notv[word]
        else:
            p_v_n = 0
            log_v_n = 0

        if word in p_p_notv:
            if p_p_notv[word] > 0:
                log_p_n = math.log2(p_p_notv[word])
            else: 
                log_p_n = 0
                p_p_n = p_p_notv[word]
        else:
            p_p_n = 0
            log_p_n = 0

        if word in p_s_notv:
            if p_s_notv[word] > 0:
                log_s_n = math.log2(p_s_notv[word])
            else: 
                log_s_n = 0
                p_s_n = p_s_notv[word]
        else:
            p_s_n = 0
            log_s_n = 0

        res3 = p_notv[word]*(p_cen_n*log_cen_n + 
            p_o_n*log_o_n + p_t_n*log_t_n +
            p_cp_n*log_cp_n + p_r_n* log_r_n+
            p_a_n*log_a_n + p_v_n* log_v_n+
            p_p_n*log_p_n + p_s_n*log_s_n)
        
        infGain[word] = res + res2 + res3
    print("ORDENANDO EL DICCIONARIO")
    #Ordenamos el IG de mayor a menor:
    infGain_sort = sorted(infGain.items(), key = operator.itemgetter(1), reverse=True)
    i = 0
    print("SACANDO LOS RESULTADOS AL FICHERO EXTERNO")
    with open('resultadosIG_test.txt', 'w') as f:
        for word in infGain_sort:
            if i <= 32768:
                w0 = str(word[0])
                w1 = str(word[1])
                s = str(w0 + ' ' + w1 + '\n')
                f.write(s)
            else: break
            i += 1

