import numpy as np
from sklearn.cluster import KMeans
import math


class XmeansObj():
    def __init__(self, X):
        self.data = X
        self.labels_ = None
        self.centroid_centres_ = None
        self.k = None
        self.count = None


def sum_euclid_dist_sq(points, centre, weights=None):
    if weights is None:
        weights = np.array([1] * len(points[0]))
    dist = np.sum(np.sqrt(np.sum(np.power(weights * (points - centre), 2), axis=1)))
    return dist


def BIC(clusters, centres, weights=None):
    scores = np.array([float('inf')] * len(clusters))

    k = len(clusters)
    m = np.size(clusters[0], axis=1)
    r = 0.0

    # calculate variance (sigma**2)
    sig_sq = 0.0
    for i in range(0, len(clusters)):
        sig_sq += sum_euclid_dist_sq(clusters[i], centres[i], weights)
        r += len(clusters[i])

    p = (k - 1) + m * k + 1

    if sig_sq <= 0:
        # in case of same points, sigma_sq can be zero
        sigma_multiplier = float('inf')
    else:
        sigma_multiplier = m * 0.5 * math.log(sig_sq)

    # calculate splitting criterion
    for i in range(0, len(clusters)):
        rn = len(clusters[i])

        # calculate log likelihood
        l1 = rn * math.log(rn)
        l2 = -rn * math.log(r)
        l3 = -rn * 0.5 * math.log(2 * np.pi)
        l4 = -rn * sigma_multiplier
        l5 = -(rn - k) * 0.5
        l = l1 + l2 + l3 + l4 + l5
        scores[i] = l
    bic = sum(scores) - p * 0.5 * math.log(r)

    return bic


def farthestPoint(array, point):
        dist = np.sqrt(np.sum((array - point) ** 2, 1))
        idx = dist.argmax()
        farthest_pt = array[idx]

        return farthest_pt, dist[idx]


def splitCluster(data, cluster, centre):
    point_array = data[cluster]
    farthest_pt, distance = farthestPoint(point_array,centre)

    vector1 = (farthest_pt - centre)/float(distance)
    vector2 = vector1*(-1)

    new_centre1 = vector1 * distance/float(3) + centre
    new_centre2 = vector2 * distance/float(3) + centre

    new_centres = np.vstack((new_centre1,new_centre2))

    new_clusters = [[], []]

    for i in range(0, len(point_array)):
        point = point_array[i]
        dist = np.sqrt(np.sum((new_centres - point) ** 2, 1))
        cluster_idx = dist.argmin()
        new_clusters[cluster_idx].append(cluster[i])

    new_centres[0] = np.mean(data[new_clusters[0]], axis=0)
    new_centres[1] = np.mean(data[new_clusters[1]], axis=0)

    return new_clusters, new_centres


def fit(data, kmax=20, kmin=2, weights=None):

    xmeans_data = XmeansObj(data)

    data_idx = np.arange(0, len(data))
    clusters = np.array([np.arange(0, len(data))])
    centres = [np.mean(data, axis=0)]

    k = kmin

    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        centres = kmeans.cluster_centers_
        new_clusters = []
        for i in range(0, kmin):
            new_clusters.append(clusters[0][labels == i])
        clusters = new_clusters

    # initially only one cluster
    n_free_centres = kmax - len(clusters)  # number of new centroids that can be added

    change = 1

    while (len(clusters) < kmax) & (change == 1):
        change = 0
        new_clusters = []
        new_centres = []
        for i in range(0, len(clusters)):
            parent_cluster = clusters[i]
            parent_centre = centres[i]
            if len(parent_cluster) > 1:
                # split cluster
                child_clusters, child_centres = splitCluster(data, parent_cluster, parent_centre)
                child_cluster1 = child_clusters[0]
                child_cluster2 = child_clusters[1]

                # calculate BIC
                parent_bic = BIC([data[parent_cluster, :]], [parent_centre], weights)
                children_bic = BIC([data[child_cluster1, :], data[child_cluster2, :]], child_centres, weights)
                if (children_bic > parent_bic) & (n_free_centres > 0):
                    new_clusters.append(child_cluster1)
                    new_clusters.append(child_cluster2)
                    new_centres.append(child_centres[0])
                    new_centres.append(child_centres[1])
                    n_free_centres -= 1
                    change = 1
                else:
                    new_clusters.append(parent_cluster)
                    new_centres.append(parent_centre)
            else:
                new_clusters.append(parent_cluster)
                new_centres.append(parent_centre)

        # update cluster set
        clusters = list(new_clusters)
        centres = list(new_centres)

    if weights is None:
        kmeans = KMeans(n_clusters=len(clusters), random_state=0).fit(data)
    else:
        kmeans = KMeans(n_clusters=len(clusters), random_state=0).fit(weights*data)
    labels = kmeans.labels_
    centres = kmeans.cluster_centers_
    new_clusters = []
    for i in range(0, len(clusters)):
        new_clusters.append(data_idx[labels == i])
    clusters = new_clusters

    # assign cluster labels for data and count number of elements in each cluster
    labels = np.array([-1] * len(data))
    count = np.array([-1] * len(clusters))
    for i in range(0, len(clusters)):
        labels[clusters[i]] = i
        count[i] = len(clusters[i])

    xmeans_data.labels_ = labels
    xmeans_data.centroid_centres_ = np.asarray(centres)
    xmeans_data.k = len(clusters)
    xmeans_data.count = count

    return xmeans_data
