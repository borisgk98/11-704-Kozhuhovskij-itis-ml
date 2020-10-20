import numpy as np
import random
from borisgk98ml.tools import *


def compute_assignments(probabilities):
    clusters_result = [0] * n
    for i in range(n):
        best_match = max(probabilities[i])
        for j in range(k):
            if probabilities[i][j] == best_match:
                clusters_result[i] = j
    return clusters_result


def compute_centers(points, probabilities):
    c = [[0, 0] for i in range(k)]

    for i in range(k):
        n = 0
        sums = [0 for i in range(dim)]

        for j in range(len(points)):
            mx = max(probabilities[j])
            if probabilities[j, i] == mx:
                n = n + probabilities[j, i] ** extra_weight
                for d in range(dim):
                    sums[d] += probabilities[j, i] ** extra_weight * points[j][d]

        if n != 0:
            for d in range(dim):
                c[i][d] = sums[d] / n
        else:
            for d in range(dim):
                c[i][d] = 0

    return c


def calculate_probabilities(n, k, points, centers):
    matrix = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            sum = 0
            dist_j = distance(points[i], centers[j])
            for t in range(k):
                dist_t = distance(points[i], centers[t])
                sum += (dist_j / dist_t) ** (2 / (extra_weight - 1))
            matrix[i, j] = 1 / sum
    return matrix


def is_precised(old_probabilities, new_probabilities):
    max = 0
    for i in range(n):
        for j in range(k):
            diff = np.abs(new_probabilities[i, j] - old_probabilities[i, j])
            if diff > max:
                max = diff
    return max < eps


def c_means(points):

    probabilities = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            probabilities[i, j] = random.randint(1, 4)

    b = True
    while b:
        centers = compute_centers(points, probabilities)
        matrix = calculate_probabilities(n, k, points, centers)
        if is_precised(matrix, probabilities):
            assignments = compute_assignments(matrix)
            show(points, assignments, centers)
            b = False
        probabilities = matrix

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

n, k, dim, extra_weight, eps = len(points), 4, len(points[0]), 1.5, 0.1

c_means(points)