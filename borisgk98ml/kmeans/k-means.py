from random import randrange
from borisgk98ml.tools import *


def point_avg(points):
    """
    расчет центра для множества точек
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    обновление центров исходя из данных и массива отношений данных ко множествам
    """
    new_means = {}
    centers = []
    for assignment, point in zip(assignments, data_set):
        if assignment not in new_means:
            new_means[assignment] = []
        new_means[assignment].append(point)

    for x in list(new_means):
        centers.append(point_avg(new_means[x]))

    return centers


def assign_points(data_points, centers):
    """
    определение точек множествам исходя из центров этих множеств
    """
    assignments = []
    for point in data_points:
        shortest = None
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if shortest is None or val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def generate_k(data_set, k):
    """
    Генерация случайных точек для старта алгоритма
    """
    centers = []
    dimensions = len(data_set[0])
    mins = []
    maxs = []

    for i in range(dimensions):
        mins.append(min(map(lambda v: v[i], data_set)))
        maxs.append(max(map(lambda v: v[i], data_set)))

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            rand_point.append(randrange(mins[i], maxs[i] + 1))
        centers.append(rand_point)

    return centers

def k_means(dataset, k):
    # стартовые центры
    centers = generate_k(dataset, k)
    assignments = assign_points(dataset, centers)
    old_assignments = None
    while assignments != old_assignments:
        show(dataset, assignments, centers)
        centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, centers)

    res = (dataset, assignments, centers)
    return show(dataset, assignments, centers)

points = [
    [1, 2],
    [2, 1],
    [3, 1],
    [5, 4],
    [5, 5],
    [6, 5],
    [10, 8],
    [7, 9],
    [11, 5],
    [14, 9],
    [14, 14],
    ]
k_means(points, 3)