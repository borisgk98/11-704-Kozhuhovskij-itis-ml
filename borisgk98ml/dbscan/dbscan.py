from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy
from borisgk98ml.tools import show

def dbscan(dataset, eps, min_ne):
    assigments = [0] * len(dataset)
    c = 0
    for point in range(0, len(dataset)):
        if not (assigments[point] == 0):
            continue

        neighbor_pts = region_query(dataset, point, eps)

        if len(neighbor_pts) < min_ne:
            # выброс
            assigments[point] = -1
        else:
            c += 1
            grow_cluster(dataset, assigments, point, neighbor_pts, c, eps, min_ne)

    return assigments


def grow_cluster(dataset, labels, point, neighbor_pts, c, eps, min_ne):
    labels[point] = c

    i = 0
    while i < len(neighbor_pts):
        pn = neighbor_pts[i]

        if labels[pn] == -1:
            labels[pn] = c

        elif labels[pn] == 0:
            labels[pn] = c

            pn_neighbor_pts = region_query(dataset, pn, eps)

            if len(pn_neighbor_pts) >= min_ne:
                neighbor_pts = neighbor_pts + pn_neighbor_pts
        i += 1


def region_query(D, point, eps):
    neighbors = []

    for Pn in range(0, len(D)):

        if numpy.linalg.norm(D[point] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors


centers = [[1, 1], [-1, -1], [1, -1]]
X, labels = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

X = StandardScaler().fit_transform(X)

# show start data
show(X, labels, [])

# show result
assignments = dbscan(X, 0.3, 10)
show(X, assignments, [])

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_assignments = db.labels_
show(X, skl_assignments, [])