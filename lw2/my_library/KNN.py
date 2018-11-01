import pandas as pd
import numpy as np
import math
import operator
from operator import itemgetter
from scipy.spatial import distance



class KNNclassifier:
    def __init__(self, data, test, k):
        self.data = data
        self.test = test
        self.k = k
    # Определение функции, которая вычисляет эвклидовое расстояние между двумя точками данных

    def euclideandistance(self, data1, data2, length):

        distance = 0
        for x in range(length):
            distance += np.square(data1[x] - data2[x])
        return np.sqrt(distance)

    # Описание KNN модели
    def knn(self, trainingSet, testInstance, k):

        distances = {}
        sort = {}

        length = testInstance.shape[1]

        #### Начало этапа 3
        # Расчет эвклидовой дистанции между каждой строкой данных обучения и данными теста
        for x in range(len(trainingSet)):
            #### Начало этапа 3.1
            dist = self.euclideandistance(testInstance, trainingSet.iloc[x], length)

            distances[x] = dist[0]
            #### Конец этапа 3.1

        #### Начало этапа 3.2
        # Сортировка их на основе расстояния
        sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
        #### Конец этапа 3.2

        neighbors = []

        #### Начало этапа 3.3
        # Extracting top k neighbors
        for x in range(k):
            neighbors.append(sorted_d[x][0])
        #### Конец этапа 3.3
        classVotes = {}

        #### Начало этапа 3.4
        # Вычисление самого частого класса среди соседей
        for x in range(len(neighbors)):
            response = trainingSet.iloc[neighbors[x]][-1]

            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        #### Конец этапа 3.4

        #### Начало этапа 3.5
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return (sortedVotes[0][0], neighbors)
        #### Конец этапа 3.5
