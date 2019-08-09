import numpy as np
from timeit import default_timer as timer
import XMeans


def computeSimilarity(q, xmeans, q_home=None, weights=None):
    # computes similarity index between configuration q and all other configurations by finding
    # the mean of the euclidean distances between q and the cluster centres of all data points.
    # If a home configuration is given this value is biased with a small contribution from the
    # distance to the home configuration.

    centroids = xmeans.centroid_centres_
    cluster_count = xmeans.count
    n_configs = sum(cluster_count)

    if weights is None:
        deltas = centroids[:, 0:6] - q[0:6]
        home_D = (sum(np.power(q[0:6] - q_home, 2)))
    else:
        deltas = weights * (centroids[:, 0:6] - q[0:6])
        home_D = (sum(np.power(weights * (q[0:6] - q_home), 2)))

    dist_2 = (np.einsum('ij,ij->i', deltas, deltas))

    weighted_avg = np.sum(cluster_count * dist_2) / n_configs

    if q_home is None:
        S = weighted_avg
    else:
        S = 0.9 * weighted_avg + 0.1 * home_D

    return S


def moveToEnd(row, q_list):
    # move configuration in row 'row' to end of q_list
    row = int(row)
    if row == 0:
        new_list = np.vstack((q_list[1:], q_list[0]))
    elif row < len(q_list) - 1:
        new_list = np.vstack((q_list[0:row], q_list[row + 1:]))
        new_list = np.vstack((new_list, q_list[row]))
    else:
        new_list = q_list

    return new_list


def moveRowsToEnd(rows, q_list):
    extract = np.array(q_list[rows])
    q_list = np.delete(q_list, rows, axis=0)
    q_list = np.vstack((q_list, extract))

    return q_list


def NearestNode(node, nodes):
    # finds closest node in cluster 'nodes' to configuration 'node'
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    idx = np.argpartition(dist_2, 2)[0:2]
    vals = dist_2[idx][0:2]
    return [idx[0], vals[0]], [idx[1], vals[1]]


def clusterConfigSelection(configurations, qhome, weights, weights_bic=None, kmin=8):
    select_start = timer()
    N_poses = int(max(configurations[:, 10])) + 1

    ignore_counter = 0
    change = 1
    threshold = 0.5

    if N_poses > 2:
        while change == 1:
            change = 0

            if ignore_counter == 0:
                q_list = np.array(configurations)
            else:
                q_list = np.array(configurations[:ignore_counter * -1, :])

            xmeans = XMeans.fit(q_list[:, 0:6], kmax=100, kmin=kmin, weights=weights_bic)

            for i in xrange(1, N_poses):
                rows = np.where(q_list[:, 10] == i)[0]
                rows = q_list[rows, 11]
                similarity = np.array([[float(1000), 1000]] * len(rows))
                row_len = len(rows)

                # Skip if only 1 configuration or none exists
                if row_len <= 1:
                    continue

                # calculate similarity heuristic
                for j in xrange(0, row_len):
                    row = np.where(q_list[:, 11] == rows[j])[0][0]
                    similarity[j] = np.append(rows[j], computeSimilarity(q_list[row], xmeans, qhome, weights))

                # sort rows in ascending order based on similarity index
                ind = np.argsort(similarity[:, 1])
                similarity = similarity[ind]

                if row_len > 2:
                    ignore_idx = max(2, round(np.log(row_len)))
                    ignore = np.array(similarity[int(ignore_idx):])
                    change = 1

                else:
                    ignore = np.array(similarity[1:])
                    change = 1

                ignore_shape = np.shape(ignore)
                if len(ignore_shape) == 2:
                    rows_to_move = []

                    for l in xrange(0, len(ignore[:, 0])):
                        if ignore[l, 1] - similarity[0, 1] > threshold:
                            ind = int(ignore[l, 0])
                            row = np.where(configurations[:, 11] == ind)[0]
                            rows_to_move.append(int(row))
                            ignore_counter += 1
                        else:
                            threshold -= 0.002

                    configurations = moveRowsToEnd(rows_to_move, configurations)

                elif ignore[1] - similarity[0, 1] > threshold:
                    configurations = moveToEnd(ignore[0], configurations)
                    ignore_counter += 1
                else:
                    threshold -= 0.002

        selected_configurations = np.array(configurations[:ignore_counter * -1, :])

    else:
        dist_2_home = np.sqrt(np.sum((configurations[:, 0:6] - qhome) ** 2, axis=1))
        best_conf = np.argmin(dist_2_home)
        selected_configurations = configurations[best_conf]

    select_time = timer() - select_start

    return selected_configurations, select_time