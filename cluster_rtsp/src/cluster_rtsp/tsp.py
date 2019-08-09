import numpy as np
import networkx as nx
from itertools import combinations


# TSP functions
def rearrangeTour(route, start, end):
    if start is not None:
        if end is not None:
            end_val = route[end]
        route = route[start:] + route[0:start]

    if end is not None:
        end_idx = route.index(end_val)
        if end_idx < len(route) - 1:
            route = route[0:end_idx] + route[end_idx + 1:] + [route[end_idx]]
    return route


def compute_tour_cost(graph, tour, weight='weight', is_cycle=True):
    cost = 0
    start = 0 if is_cycle else 1
    for idx in xrange(start, len(tour)):
        u = tour[idx - 1]
        v = tour[idx]
        cost += graph.edge[u][v][weight]
    return cost


def euclidean_fn(x, y, weights=None):
    if weights is None:
        distance = np.sqrt(np.sum((x - y) ** 2))
    else:
        distance = np.sqrt(np.sum(weights * (x - y) ** 2))
    return distance


def construct_tgraph(coordinates, distfn=None, args=()):
    if distfn is None:
        distfn = euclidean_fn
    num_nodes = len(coordinates)
    graph = nx.Graph()
    for i in xrange(num_nodes):
        for j in xrange(i + 1, num_nodes):
            graph.add_node(i, value=coordinates[i])
            graph.add_node(j, value=coordinates[j])
            dist = distfn(coordinates[i], coordinates[j], *args)
            graph.add_edge(i, j, weight=dist)
    return graph


def two_opt(graph, weight='weight', start=None, end=None):
    num_nodes = graph.number_of_nodes()
    tour = graph.nodes()
    if (start is not None) | (end is not None):
        if start is None:
            start = 0
            begin_i = 0
        else:
            begin_i = 1
        if end is None:
            end = num_nodes - 1
            end_j = num_nodes
        else:
            end_j = num_nodes - 1
        tour = rearrangeTour(tour, start, end)
    else:
        begin_i = 1
        end_j = num_nodes
    start_again = True
    loop_n = 0
    while start_again:
        loop_n += 1
        start_again = False
        for i in xrange(begin_i, end_j - 1):
            for k in xrange(i + 1, end_j):
                # 2-opt swap
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % num_nodes]
                if (a == c) or (b == d):
                    continue
                ab_cd_dist = graph.edge[a][b][weight] + graph.edge[c][d][weight]
                ac_bd_dist = graph.edge[a][c][weight] + graph.edge[b][d][weight]
                if ab_cd_dist > ac_bd_dist:
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    start_again = True
                if start_again:
                    break
            if start_again:
                break
    min_cost = compute_tour_cost(graph, tour)
    return tour


## Global TSP functions
def minEuclid(point, array):

    if len(np.shape(array)) > 1:
        dist = np.sqrt(np.sum((array - point) ** 2, axis=1))
        return np.argmin(dist), min(dist)
    else:
        dist = np.sqrt(np.sum((array - point) ** 2))
        return 0, dist


def dsearchn(point, array, ignore=None):
    # find nearest element to point in array

    if (len(array) == 1) & (ignore is not None):
        return None

    dist = np.sqrt(np.sum((array - point) ** 2, 1))
    if ignore is not None:
        dist[ignore] = 10000
    idx = dist.argmin()
    return idx


def closestPoints(clusters, pairs):
    closest_points = np.array([[0.0] * 3] * len(pairs))
    for i in xrange(0, len(pairs)):
        try:
            cluster1 = clusters[pairs[i, 0]][:, 0:6]
        except IndexError:
            cluster1 = clusters[pairs[i, 0]][0:6]
        try:
            cluster2 = clusters[pairs[i, 1]][:, 0:6]
        except IndexError:
            cluster2 = clusters[pairs[i, 1]][0:6]
        min_dist = float('inf')
        for idx1 in xrange(0, len(cluster1)):
            point = cluster1[idx1]
            idx2, dist = minEuclid(point, cluster2)
            if dist < min_dist:
                best_idx1 = idx1
                best_idx2 = idx2
                min_dist = dist
        closest_points[i, 0:3] = [best_idx1, best_idx2, min_dist]

    return closest_points


def globalTSP(clusters, home_pose):
    # add home pose as new start and end clusters
    clusters.append(np.hstack((home_pose, [0, 0, 0, 0, 0, 0])))
    clusters.append(np.hstack((home_pose, [0, 0, 0, 0, 0, 0])))
    N = len(clusters)

    # get cluster pairings
    combs = combinations(range(0, N), 2)
    pairs = np.array(list(combs))

    # find best distance
    closest_points = closestPoints(clusters, pairs)

    # create graph
    graph = nx.Graph()
    for i in range(0, N):
        graph.add_node(i)
    for i in range(0, len(pairs)):
        graph.add_edge(pairs[i, 0], pairs[i, 1], weight=closest_points[i, 2])

    # solve TSP
    gtour = two_opt(graph, start=N - 2, end=N - 1)

    entry_points = np.array([[0, 0]] * N)
    # extract entry and exit points
    for i in range(1, len(gtour) - 1):
        before = gtour[i - 1]
        current = gtour[i]
        after = gtour[i + 1]
        if before < current:
            entry_idx = np.where((pairs[:, 0] == before) & (pairs[:, 1] == current))[0]
            entry_points[i][0] = closest_points[entry_idx, 1]
        else:
            entry_idx = np.where((pairs[:, 0] == current) & (pairs[:, 1] == before))[0]
            entry_points[i][0] = closest_points[entry_idx, 0]
        if current < after:
            exit_idx = np.where((pairs[:, 0] == current) & (pairs[:, 1] == after))[0]
            exit_pt = closest_points[exit_idx, 0]
            next_entry = closest_points[exit_idx, 1]

        else:
            exit_idx = np.where((pairs[:, 0] == after) & (pairs[:, 1] == current))[0]
            exit_pt = closest_points[exit_idx, 1]
            next_entry = closest_points[exit_idx, 0]

        if exit_pt != entry_points[i][0]:
            entry_points[i][1] = exit_pt
        else:
            try:
                exit_pt = dsearchn(clusters[after][int(next_entry), 0:6], clusters[current][:, 0:6],
                                   ignore=[int(exit_pt)])
            except IndexError:
                exit_pt = dsearchn(clusters[after][0:6], clusters[current][:, 0:6], ignore=[int(exit_pt)])
            entry_points[i][1] = exit_pt

    return gtour, pairs, closest_points, entry_points
