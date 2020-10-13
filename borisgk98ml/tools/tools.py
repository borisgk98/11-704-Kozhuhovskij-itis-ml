import matplotlib.pyplot as plt
import numpy as np

colors = ["red", "green", "blue", "purple", "pink", "yellow", "gray", "orange"]
def show(dataset, assignments, centers):

    dimensions = len(dataset[0])
    if dimensions != 2:
        return "Cannot show plot"

    d = dict(zip(assignments, dataset))

    for i in range(len(dataset)):
        assignment = assignments[i]
        if type(assignment) != int or assignment >= 0:
            plt.scatter(dataset[i][0], dataset[i][1], color=colors[assignment])
        else:
            plt.scatter(dataset[i][0], dataset[i][1], color='black')

    for c in centers:
        plt.scatter(c[0], c[1], color='black')

    plt.show()


def distance(a, b):
    """
    расчет дистанции
    """
    dim = len(a)
    _sum = 0
    for dimension in range(dim):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return np.sqrt(_sum)