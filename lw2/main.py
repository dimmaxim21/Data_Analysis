"""
Write a program
"""
#

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import math
import operator
from my_library import KNN


data = pd.read_csv('K:/iris-label.csv')
data.head()

# Создание набора фиктивных тестов
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

print('\ntest------------------------------------\n')

#### Начало этапа 2
# Установка количества соседей = 1
k = 1
#### Конец этапа 2
# Running KNN model
knn_classifier = KNN.KNNclassifier(data, test, k)
result, neigh = knn_classifier.knn(data, test, k)
# Predicted class
print(result)
print(neigh)


# Установка количества соседей = 3
k = 3
# Запуск KNN модели
result, neigh = knn_classifier.knn(data, test, k)
# Прогнозируемый класс
print('\n')
print(result)
print(neigh)


# Установка количества соседей = 5
k = 5
# Запуск KNN модели
result, neigh = knn_classifier.knn(data, test, k)
# Прогнозируемый класс
print('\n')
print(result)
print(neigh)


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data.iloc[:,0:4], data['Name'])

# Прогнозируемый класс
print('\n')
print(neigh.predict(test))
# 3 ближайших соседей
print(neigh.kneighbors(test)[1])


feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
X = data[feature_names]
y = data['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 1
result,neigh = knn_classifier.knn(data, X_test, k)
print('\nrun-------------------------------------\n')
print(result)
print(neigh)

k = 3
result,neigh = knn_classifier.knn(data, X_test, k)
print('\n')
print(result)
print(neigh)

k = 5
result,neigh = knn_classifier.knn(data, X_test, k)
print('\n')
print(result)
print(neigh)