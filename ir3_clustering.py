"""
This module defines evaluation for
document clustering using kMeans.
It clusters documents together based
on similarity measure with k centroids.


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

from collections import Counter

import numpy as np
import pandas as pd


def evaluate_clustering(
        vsm_matrix: pd.DataFrame,
        k=5,
        store_results=False
):
    # select k centroids at random
    centroids = pd.DataFrame.copy(vsm_matrix.sample(n=k), deep=True)
    centroids.reset_index(drop=True, inplace=True)

    clusters = {}  # k clusters [0.. k-1]

    iterations = 0
    while True:  # centroids are changing
        iterations += 1
        similarity_matrix = _measure_cos_sim(centroids, vsm_matrix)
        cluster_vector = _create_cluster_vector(similarity_matrix)

        # map documents to clusters {cluster_id: [doc_id1 .. doc_idk]}
        clusters = cluster_vector.groupby(cluster_vector).groups
        clusters = {idx: vsm_matrix.loc[indices] for idx, indices in clusters.items()}

        old_centroids = centroids
        centroids = _calculate_centroids(clusters)

        if centroids.equals(old_centroids):
            break  # stop if centroids and clusters are not changing

    results_dict = {idx: vectors.index for idx, vectors in clusters.items()}
    results = pd.DataFrame.from_dict(results_dict, orient='index')

    # Cluster X Category Matrix used for purity measure
    summary = {}
    for idx, doc_ids in results_dict.items():
        summary[idx] = [doc_id.rstrip('1234567890_') for doc_id in doc_ids]

    summary = {cluster_id: Counter(categories) for cluster_id, categories in summary.items()}
    summary = pd.DataFrame.from_dict(data=summary, orient='index')
    summary.fillna(0, inplace=True)

    if store_results is True:
        results.to_csv('q2_results.csv')
        summary.to_csv('q2_summary.csv')

    # purity = 1/N * (sum of count of most common class in each cluster)
    purity = summary.max(axis=1).sum()
    purity /= summary.values.sum()

    return purity


def _measure_cos_sim(columns_set: pd.DataFrame, rows_set: pd.DataFrame):
    similarity_matrix = rows_set.dot(columns_set.transpose())
    return similarity_matrix


def _create_cluster_vector(similarity_matrix: pd.DataFrame) -> pd.Series:
    return similarity_matrix.idxmax(axis=1)  # max because similarity measure


def _calculate_centroids(clusters: dict):
    # average of all vectors within the cluster

    #                               | collapse index
    centroids = {idx: vectors.mean(axis=0) for idx, vectors in clusters.items()}
    centroids = pd.DataFrame.from_dict(centroids, orient='index')

    # normalize
    row_magnitudes = np.sqrt(np.square(centroids).sum(axis=1))
    centroids = centroids.div(row_magnitudes, axis=0)

    return centroids
