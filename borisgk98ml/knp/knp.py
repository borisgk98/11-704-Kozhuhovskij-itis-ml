from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from borisgk98ml.tools import *


# для построения минимального остовного дерева используется алгоритм прима с асимптотикой О(n^2)
# https://e-maxx.ru/algo/mst_prim#6
def prima(matrix, inf=None):
    res = []

    n = len(matrix)
    used = [False] * n
    min_e = [inf] * n
    sel_e = [-1] * n

    for i in range(n):
        v = -1
        for j in range(n):
            if used[j] is False and (v == -1 or min_e[j] < min_e[v]):
                v = j

        used[v] = True
        if sel_e[v] != -1:
            res.append((v, sel_e[v]))

        for to in range(n):
            if matrix[v][to] < min_e[to]:
                min_e[to] = matrix[v][to]
                sel_e[to] = v

    return res

# Используем систему непересекающихся множеств, чтобы построить искомые множества
# https://e-maxx.ru/algo/dsu
def dsu(n, vers):
    parents = list(range(n))
    # для ранговой эвристики
    rangs = [0] * n

    def merge(u1, u2):
        if rangs[u1] > rangs[u2]:
            parents[u2] = parents[u1]
            rangs[u1] += rangs[u2]
        else:
            parents[u1] = parents[u2]
            rangs[u2] += rangs[u1]

    def find(v):
        if parents[v] == v:
            return v
        parents[v] = find(parents[v])
        return parents[v]

    for v in vers:
        merge(v[0], v[1])

    # распределение точек по множествам (не нормализированное)
    return list(map(find, range(n)))

# сжимаем массив
def normalize(v):
    n = len(v)
    r = [-1] * n
    u = [-1] * n
    curr = 0
    for i in range(n):
        if u[v[i]] == -1:
            u[v[i]] = curr
            curr += 1
        r[i] = u[v[i]]
    return r

# Алгоритм кратчайшего незамкнутого пути
def knp(X, nclusters):
    matrix = []
    n = len(X)
    for i in range(n):
        matrix.append([])
        for j in range(n):
            matrix[i].append(distance(X[i], X[j]))

    verts = sorted(list(map(lambda v: (v[0], v[1], distance(X[v[0]], X[v[1]])), prima(matrix, 100500))), key=lambda v: v[2])
    res_verts = list(map(lambda v: (v[0], v[1]), verts[:-(nclusters - 1)]))
    return normalize(dsu(n, res_verts))

# start data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels = make_blobs(n_samples=90, centers=centers, cluster_std=0.3, random_state=0)
X = StandardScaler().fit_transform(X)

# show start data
show(X, labels, [])

n = len(centers)
assignments = knp(X, n)
show(X, assignments, [])