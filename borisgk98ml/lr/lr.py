import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from random import randrange
import numpy.linalg as linalg

n = 100
minx, maxx = 0, n
miny, maxy = 0, n
minz, maxz = 0, n
cluster_1 = []
cluster_2 = []

def get_one_dim(X, dim):
    return list(map(lambda x: x[dim], X))

# генерируем точки случайно
def generate_point():
    x = randrange(minx, maxx)
    y = randrange(miny, maxy)
    z = randrange(minz, maxz)
    return [x, y, z]

# генерируем точки случайно
def generate_points():
    return [generate_point() for _ in range(n)]

# Тестовая плоскость, нужна для разбивки данных
def get_test_plane():
    points = [np.array(generate_point()) for _ in range(3)]

    # These two vectors are in the plane
    v1 = points[2] - points[1]
    v2 = points[1] - points[0]

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    return {"a": a, "b": b, "c": c}


# разбивка на кластеры
def get_assignments(X, plane):
    clusters = []
    for x in X:
        x_i = x[0]
        y_i = x[1]
        z_i = x[2]
        if plane["a"] * x_i + plane["b"] * y_i + plane["c"] * z_i < 0:
            clusters.append(0)
            cluster_1.append(x)
        else:
            clusters.append(1)
            cluster_2.append(x)
    return clusters


# поиск точек разделяющей плоскости с помощью найденных весов для построения
def find_points_for_dividing_plane(model):
    plane_z = np.ones((100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            plane_z[i][j] = (-model.coef_[0][0] * i - model.coef_[0][1] * j - model.intercept_) / model.coef_[0][2]
    return plane_z


# визуализация
def show(plane):
    fig = go.Figure(data=[
        go.Scatter3d(x=get_one_dim(cluster_1, 0), y=get_one_dim(cluster_1, 1), z=get_one_dim(cluster_1, 2), mode="markers", name="-1"),
        go.Scatter3d(x=get_one_dim(cluster_2, 0), y=get_one_dim(cluster_2, 1), z=get_one_dim(cluster_2, 2), mode="markers", name="1"),
        go.Surface(z=plane)
    ])
    fig.show()


def lr():
    X = generate_points()
    plane = get_test_plane()
    y = get_assignments(X, plane)
    model = LogisticRegression().fit(X, y)
    plane_z = find_points_for_dividing_plane(model)
    show(plane_z)


lr()
