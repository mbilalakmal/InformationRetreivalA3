"""
This module defines a Vector Space Model
for documents and features.
It creates a vsm matrix from documents
and features along with a response vector
which stores the correct category for
each document (class labels).


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

from collections import Counter

import numpy as np
import pandas as pd


def create_vsm_matrix(documents: dict):

    # count frequencies of features in every document
    term_frequencies = {doc_id: Counter(doc.terms) for doc_id, doc in documents.items()}

    # create vsm_matrix (DataFrame) and response_vector (Series)
    vsm_matrix = pd.DataFrame.from_dict(data=term_frequencies, orient='index')
    vsm_matrix.fillna(0, inplace=True)

    responses = {doc.doc_id: doc.category for doc in documents.values()}
    response_vector = pd.Series(responses)

    # apply feature scaling (TF-IDF)
    vsm_matrix = _apply_tfidf(vsm_matrix)

    # apply normalization (unit vectors)
    vsm_matrix = _apply_normalization(vsm_matrix)

    return vsm_matrix, response_vector


def _apply_tfidf(vsm_matrix):

    # sum nonzero values along vertical axis
    document_frequencies = (vsm_matrix != 0).sum(0)

    idf_vector = np.log(vsm_matrix.shape[0]/document_frequencies)

    vsm_matrix += 1
    vsm_matrix = np.log(vsm_matrix)

    vsm_matrix *= idf_vector

    return vsm_matrix


def _apply_normalization(vsm_matrix):
    row_magnitudes = np.sqrt(
        np.square(vsm_matrix).sum(axis=1)
    )
    vsm_matrix = vsm_matrix.div(row_magnitudes, axis=0)

    return vsm_matrix
