"""
This module defines evaluation for
document classification using kNN.
It splits the data set into test and train
and predicts the categories of test set
using a similarity measure with train set.


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

import pandas as pd


def evaluate_classification(
        vsm_matrix: pd.DataFrame,
        response_vector: pd.Series,
        k=3,
        ratio=0.7,
        store_results=False
):
    train_set = vsm_matrix.sample(frac=ratio)
    test_set = vsm_matrix.drop(train_set.index)

    # calculate similarity scores of test set with train set
    similarity_matrix = _measure_cos_sim(train_set, test_set)
    # select k largest scores (kNN)
    knn_matrix = _create_knn_matrix(similarity_matrix, response_vector, k)

    # predict categories vector (Series) using most repeated category
    predict_vector = knn_matrix.mode(axis='columns')[0]

    # store results
    if store_results:
        predict_vector.to_csv('q1_results.csv')

    # bool list containing True for accurate classification
    accurate = [response_vector[doc_id] == category for doc_id, category in predict_vector.iteritems()]
    accuracy = (sum(accurate) / len(accurate))

    return accuracy


def _measure_cos_sim(train_set: pd.DataFrame, test_set: pd.DataFrame):
    similarity_matrix = test_set.dot(train_set.transpose())
    return similarity_matrix


def _create_knn_matrix(similarity_matrix, response_vector, k):
    k_closest_dict = {
        doc_id: row.nlargest(k).index.tolist() for doc_id, row in similarity_matrix.iterrows()
    }

    k_closest_matrix = pd.DataFrame.from_dict(k_closest_dict, orient='index')
    # |
    # | this matrix contains k documents
    # | from train set (columns) for
    # | each document from test set (index)

    # like k_closest_matrix but replaces train set
    # document ids with their category label
    knn_matrix = k_closest_matrix.applymap(
        lambda doc_id: response_vector[doc_id]
    )
    return knn_matrix
