from sklearn.datasets import make_blobs
from borisgk98ml.tools import *
import random
from math import sqrt
from math import floor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Train data generator
def generateData(number_of_class_el, number_of_classes):
    data = []
    classes = []
    for classNum in range(number_of_classes):
        centerX, centerY = random.random() * 5.0, random.random() * 5.0
        for rowNum in range(number_of_class_el):
            data.append([random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)])
            classes.append(classNum)
    return (data, classes)

def knn_single(X, y, number_of_classes, point):
    v = [0] * number_of_classes
    n = len(X)
    k = floor(sqrt(n))
    nn = []
    for i in range(n):
        nn.append((y[i], distance(X[i], point)))
    nn = sorted(nn, key=lambda x: x[1])
    knn = nn[:k]
    c = None
    cm = 0
    for i in range(k):
        v[knn[i][0]] += 1
        if v[knn[i][0]] > cm:
            cm = v[knn[i][0]]
            c = knn[i][0]

    return c

def knn(X, y, number_of_classes, points):
    r = []
    for point in points:
        r.append(knn_single(X, y, number_of_classes, point))
    return r

number_of_class_el = 700
number_of_classes = 10

X, y = generateData(number_of_class_el, number_of_classes)
# show(X, y, [])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# show(X_train, y_train, [])

# my knn
y_pred = knn(X_train, y_train, number_of_classes, X_test)
# show(X_train + X_test, y_train + y_pred, [])
print(accuracy_score(y_test, y_pred))

# skl knn
skl_knn = KNeighborsClassifier(n_neighbors=floor(sqrt(len(X))), metric="euclidean", weights="distance").fit(X_train, y_train)
y_pred = skl_knn.predict(X_test).tolist()
# show(X_train + X_test, y_train + y_pred, [])
print(accuracy_score(y_test, y_pred))

