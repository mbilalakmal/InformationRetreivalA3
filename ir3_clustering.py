"""


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

from collections import Counter

import numpy as np
import pandas as pd


def evaluate_clustering(
        vsm_matrix: pd.DataFrame,
        # response_vector: pd.Series,
        k=5,
        store_results=False
):
    # k centroids at random
    centroids = pd.DataFrame.copy(vsm_matrix.sample(n=k), deep=True)
    centroids.reset_index(drop=True, inplace=True)

    # print(centroids)

    clusters = {}

    iterations = 0

    while (
        # centroids not changing
        True
    ):
        similarity_matrix = _measure_cos_sim(centroids, vsm_matrix)
        cluster_vector = _create_cluster_vector(similarity_matrix)

        clusters = cluster_vector.groupby(cluster_vector).groups
        clusters = {idx: vsm_matrix.loc[indices] for idx, indices in clusters.items()}

        old_centroids = centroids
        centroids = _calculate_centroids(clusters)

        iterations += 1

        if centroids.equals(old_centroids):
            break

    for idx, vectors in clusters.items():
        print(f'Cluster{idx} has {len(vectors)} documents')

    results_dict = {idx: vectors.index for idx, vectors in clusters.items()}
    results = pd.DataFrame.from_dict(results_dict, orient='index')

    summary = {}
    for idx, doc_ids in results_dict.items():
        summary[idx] = [doc_id.rstrip('1234567890_') for doc_id in doc_ids]

    summary = {cluster_id: Counter(categories) for cluster_id, categories in summary.items()}
    summary = pd.DataFrame.from_dict(data=summary, orient='index')
    summary.fillna(0, inplace=True)

    # purity = 1/N * (sum of count of most common class in each cluster)
    purity = summary.max(axis=1).sum()
    purity /= summary.values.sum()
    print(f'Purity: {purity}')

    print(summary)

    results.to_csv('clusters.csv')
    # summary.to_csv('summary.csv')

    print(iterations)

    return purity


def _measure_cos_sim(columns_set: pd.DataFrame, rows_set: pd.DataFrame):
    similarity_matrix = rows_set.dot(columns_set.transpose())
    return similarity_matrix


def _create_cluster_vector(similarity_matrix: pd.DataFrame) -> pd.Series:
    return similarity_matrix.idxmax(axis=1)


def _calculate_centroids(clusters: dict):
    # average of all vectors within the cluster

    #                               | collapse index
    centroids = {idx: vectors.mean(axis=0) for idx, vectors in clusters.items()}

    centroids = pd.DataFrame.from_dict(centroids, orient='index')

    row_magnitudes = np.sqrt(
        np.square(centroids).sum(axis=1)
    )
    centroids = centroids.div(row_magnitudes, axis=0)

    return centroids
#
#
#
# with open(r'vsm_index', 'rb') as file:
#     vsm_index = pickle.load(file)
#
# evaluate_clustering(vsm_index[0], k=5)
