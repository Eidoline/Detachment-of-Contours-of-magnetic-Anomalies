###############################
###                         ###
###  SOME KIND OF TUTORIAL  ###
###                         ###
###############################
# Эта программа создана для выделения контуров магнитных аномалий на магнитометрических снимках.
#
# Запуск: go(file_path, contur_neighbourhood, contur_constant, initial_level,
#                       dbscan_neighbourhood, dbscan_coreneighbours, glue_eps, sift_level,
#                       smooth_radius, smooth_step, method)
# В обработке учавствуют несколько алгоритмов. В параметрах передаются все необходимые параметры используемых алгоритмов.
# 
# file_path                                   - Имя файла (jpg) обрабатываемого.
#
# contur_neighbourhood, contur_constant       - Параметры алгоритма адаптивной пороговой бинаризации: neighbourhood -
#                                               окрестность пикселя для вычисления порогового значения.
#                                               (Чем больше neighbourhood, тем толще контур.)
#                                               contrast - константа для вычисления порогового значения.
#                                               (Чем меньше constant, тем меньше деталей.)
#
# initial_level                               - Первичная фильтрация:
#                                               Минимальная площадь элементов для фильтрации сразу после адаптивной пороговой бинаризации.
#
# dbscan_neighbourhood, dbscan_coreneighbours - Параметры DBSCAN: neighbourhood - расстояние, на котором два элемента считаются соседями.
#                                                                 coreneighbours - количество необходимых соседей внутренней точки.
#
# glue_eps                                    - Параметр склеивания кластеров: на каком расстоянии должны находится кластеры (ближайщие точки),
#                                               чтобы они склеились.
#
# sift_level                                  - Параметр вторичной фильтрации: Минимальная площадь кластеров для фильтра.
#
# smooth_radius, smooth_step                  - Параметры поточечного сглаживания: radius - в какой окрестности рассматриваются соседи.
#                                                                                  step - сколько соседей одной точки должно найтись.
#                                               На данный момент сглаживание отключено (см. закомментированные строчки), так как оно не оптимизировано.
#
# method                                      - Метод обработки: 0 - сначала склеивание, потом вторичное сглаживание.
#                                                                1 - сначала вторичное сглаживание, потом склеивание.
#

#################################
###                           ###
###  THANK YOU AND GOOD LUCK  ###
###                           ###
#################################

import cv2 as cv
import pandas as pd
import random
import numpy as np
import math
import sklearn.cluster
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering 
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabaz_score

def createSamples(image):
    array = np.zeros((len(image) * len(image[0]), 3), dtype=np.float32)
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            idx = i * len(image[0]) + j
            array[idx, 0] = i
            array[idx, 1] = j
            array[idx, 2] = image[i, j]
    return array

def createSamplesFromNonZero(image):
    array = []
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if image[i, j] > 0:
                array.append((i, j))
    return np.array(array)


factor = 180

def extractResult(source, unscaledSamples, labels):
    global factor
    image = np.zeros((len(source), len(source[0]), 3), dtype=np.uint8)
    maxv = np.max(labels)    
    k = 0
    while k < len(unscaledSamples):
        i = unscaledSamples[k, 0]
        j = unscaledSamples[k, 1]
        if (labels[k]==-1):
            image[i, j, 0] = 0
            image[i, j, 1] = 0
            image[i, j, 2] = 0
        else:
            image[i, j, 0] = factor * labels[k]
            image[i, j, 1] = 255
            image[i, j, 2] = 255
        k += 1

    return cv.cvtColor(image, cv.COLOR_HSV2BGR)

def dbscan(image, dbscan_neighbourhood, dbscan_coreneighbours):
    X = createSamplesFromNonZero(image)
    df = pd.DataFrame(X)
    
    unscaledX = X
    
    db = DBSCAN(eps=dbscan_neighbourhood, min_samples=dbscan_coreneighbours, metric="euclidean").fit(df)    
    labels = db.labels_

    result = X, labels
    return result

def nothing(x):
    pass

def findExtremas(image,contur_neighbourhood,contur_constant,initial_level):
    result = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, contur_neighbourhood, contur_constant)

    labels, contours, hier = cv.findContours(result, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area < initial_level:
            cv.drawContours
            cv.drawContours(result, [contour], 0, 0, -1)

    return result

def dist(x,y):
    return ((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))**0.5

def superdbscan(X, labels, eps):
    b = True
    while b:
        b= False
        for i in set(labels):
            for j in set(labels):
                if (i>j and j>-1):
                    mm=dist(X[:][labels==i][0],X[:][labels==j][0])
                    for xi in X[:][labels==i]:
                        for xj in X[:][labels==j]:
                            if (mm > dist(xi,xj)):
                                mm = dist(xi,xj)
                    if (mm<eps):
                        labels[:][labels==i] = j
                        i=j
                        b = True
        result = X, labels
    return result

def HISTS(source, X, labels):
    global XXX
    XXX = pd.DataFrame(X)
    XXX[2] = ((np.asarray(XXX[0])-np.mean(XXX[0]))**2+(np.asarray(XXX[1])-np.mean(XXX[1]))**2)**0.5
    plt.figure(1)
    plt.subplot(222)
    plt.hist(XXX.sort_values(by=[2])[2],bins=20,rwidth=0.8)
    plt.plot([i for i in range(0,int(max(XXX[2])))],[3.141592*(i**2-(i-1)**2) for i in range(0,int(max(XXX[2])))])
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.text(max(XXX[2])/5,3.141592*((max(XXX[2])*4/5)**2-(max(XXX[2])*4/5-1)**2),str(len(XXX)))
    plt.subplot(221)
    plt.hist(XXX[0],bins=20,rwidth=0.8)
    plt.xlim(0,len(source))
    plt.subplot(224)
    plt.hist(XXX[1],bins=20,rwidth=0.8, orientation="horizontal")
    plt.ylim(0,len(source[0]))
    plt.subplot(223)
    plt.scatter(XXX[0],XXX[1],marker='.')
    plt.xlim(0,len(source))
    plt.ylim(0,len(source[0]))
    plt.show()

def sift(X, labels, level):
    for i in set(labels):
        if (len(labels[labels==i])>level):
            labels[labels==i] = i
        else:
            labels[labels==i] = -1
    result = X, labels
    return result

from collections import Counter

def smooth(source, X, labels, radius, step):
    global factor
    image = np.zeros((len(source), len(source[0]), 3), dtype=np.uint8)
    k = 0
    while k < len(X):
        i = X[k, 0]
        j = X[k, 1]
        if (labels[k]==-1):
            image[i, j, 0] = 0
            image[i, j, 1] = 0
            image[i, j, 2] = 0
        else:
            image[i, j, 0] = (labels[k]+1)*factor
            image[i, j, 1] = 255
            image[i, j, 2] = 255
        
        k += 1
        
    for i in range(len(source)):
        for j in range(len(source[0])):
            s = Counter(image[max(0,i-radius):min(len(source[0]),i+radius+1),max(0,j-radius):min(len(source[0]),j+radius+1),0].ravel()).most_common(1)
            if (len(s)==0):
                image[i,j,0] = 0
            else:
                if (s[0][1]>step):
                    image[i,j,0]=s[0][0]
            if (image[i, j, 0] == 0):
                image[i, j, 1] = 0
                image[i, j, 2] = 0
            else:
                image[i, j, 1] = 255
                image[i, j, 2] = 255
    for i in range(len(source)):
        for j in range(len(source[0])):
            if (image[i, j, 0] > 0):
                image[i, j, 0] = image[i, j, 0]-factor
    return cv.cvtColor(image, cv.COLOR_HSV2BGR)

def go(file_path, contur_neighbourhood, contur_constant, initial_level, dbscan_neighbourhood, dbscan_coreneighbours, eps, level, radius, step, method):
    print("go(\"",file_path, "\",", contur_neighbourhood,",", contur_constant,",", initial_level,",",\
          dbscan_neighbourhood,",", dbscan_coreneighbours,",", eps,",", level,",", radius,",", step,",", method,")",sep="")

    path = file_path
    

    source = cv.imread(path)
    image = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    
    cv.imshow("source", image)
    cv.moveWindow("source",0,0)

    extremas = findExtremas(image,contur_neighbourhood,contur_constant,initial_level)
    cv.imshow("extremas", extremas)
    cv.moveWindow("extremas",len(image[0]),0)

    
    dbresult = dbscan(extremas,dbscan_neighbourhood, dbscan_coreneighbours)

    global factor
    if (np.max(dbresult[1])>0):
        factor = 180 // int(np.max(dbresult[1]))
        
    cv.imshow("dbscan", extractResult(image, dbresult[0], dbresult[1]))
    cv.moveWindow("dbscan",2*len(image[0]),0)

    
    if (method==0):
        
        sdbresult = superdbscan(dbresult[0], dbresult[1], eps)        
        cv.imshow("sdbscan", extractResult(image, sdbresult[0], sdbresult[1]))
        cv.moveWindow("sdbscan",3*len(image[0]),0)
        
        siftresult = sift(sdbresult[0], sdbresult[1], level)        
        cv.imshow("siftscan", extractResult(image, siftresult[0], siftresult[1]))
        cv.moveWindow("siftscan",4*len(image[0]),0)

        #smoothresult = smooth(image, siftresult[0], siftresult[1], radius, step)
        #cv.imshow("smooth", smoothresult)
        #cv.moveWindow("smooth",5*len(image[0]),0)

    if (method==1):
            
        siftresult = sift(dbresult[0], dbresult[1], level)
        cv.imshow("siftscan", extractResult(image, siftresult[0], siftresult[1]))
        cv.moveWindow("siftscan",3*len(image[0]),0)
    
        sdbresult = superdbscan(siftresult[0], siftresult[1], eps)       
        cv.imshow("sdbscan", extractResult(image, sdbresult[0], sdbresult[1]))
        cv.moveWindow("sdbscan",4*len(image[0]),0)

        #smoothresult = smooth(image, sdbresult[0], sdbresult[1], radius, step)
        #cv.imshow("smooth", smoothresult)
        #cv.moveWindow("smooth",5*len(image[0]),0)

    smoothresult = sdbresult
    HISTS(image, smoothresult[0][smoothresult[1]>-1], smoothresult[1][smoothresult[1]>-1])

    
go("images/data 6.jpg",19,-5,30,2,3,12,100,2,16,0)



