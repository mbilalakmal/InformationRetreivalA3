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


def create_vsm_matrix(
        documents: dict,
        scaling=True,
        selection=False
    ):

    # count frequencies of features in every document
    term_frequencies = {doc_id: Counter(doc.terms) for doc_id, doc in documents.items()}

    # create vsm_matrix (DataFrame) and response_vector (Series)
    vsm_matrix = pd.DataFrame.from_dict(data=term_frequencies, orient='index')
    vsm_matrix.fillna(0, inplace=True)

    responses = {doc.doc_id: doc.category for doc in documents.values()}
    response_vector = pd.Series(responses)

    # apply feature selecetion (MI)
    if selection is True:
        vsm_matrix = _apply_mi(vsm_matrix, response_vector)

    # apply feature scaling (TF-IDF)
    if scaling is True:
        vsm_matrix = _apply_tfidf(vsm_matrix)

    # apply normalization (unit vectors)
    vsm_matrix = _apply_normalization(vsm_matrix)

    return vsm_matrix, response_vector


def _apply_tfidf(vsm_matrix):

    # sum nonzero values along vertical axis
    document_frequencies = (vsm_matrix != 0).sum(0)
    idf_vector = np.log(vsm_matrix.shape[0]/document_frequencies)

    vsm_matrix = np.log(vsm_matrix + 1)
    vsm_matrix *= idf_vector

    return vsm_matrix


def _apply_mi(vsm_matrix, response_vector):
    # total documents
    ndoc = len(response_vector)

    vsm_bool = vsm_matrix.copy(deep=True)
    vsm_bool[vsm_bool != 0] = 1
    vsm_bool['category'] = response_vector  # add category column
    confusion_matrix = vsm_bool.groupby(['category']).sum()
    confusion_matrix /= ndoc

    category_priors = response_vector.groupby(response_vector).count()
    category_priors /= ndoc  # p of doc belonging to category

    term_priors = (vsm_matrix != 0).sum(0)
    term_priors /= ndoc  # p of doc containing term

    # create 00, 01, 11, 10 matrices
    n_11 = confusion_matrix             # category = term = True
    n_00 = 1 - confusion_matrix         # category = term = False
    n_10 = confusion_matrix.rsub(term_priors, axis='columns')       # category = False term = True
    n_01 = confusion_matrix.rsub(category_priors, axis='index')       # category = True term = False

    # to avoid divide by zero error
    n_11[n_11 == 0] = 0.1/ndoc
    n_00[n_00 == 0] = 0.1 / ndoc
    n_10[n_10 == 0] = 0.1 / ndoc
    n_01[n_01 == 0] = 0.1 / ndoc

    # broadcast term_priors and category_priors
    category_priors = pd.DataFrame(category_priors)
    term_priors_t = pd.DataFrame(term_priors)
    term_priors = term_priors_t.transpose()

    # create I(U, C) matrix | mutual information
    mi_11 = n_11 * np.log2(n_11 / np.dot(category_priors, term_priors))
    mi_00 = n_00 * np.log2(n_00 / np.dot(1 - category_priors, 1 - term_priors))
    mi_10 = n_10 * np.log2(n_10 / np.dot(1 - category_priors, term_priors))
    mi_01 = n_01 * np.log2(n_01 / np.dot(category_priors, 1 - term_priors))

    mi_matrix = mi_00 + mi_01 + mi_10 + mi_11

    # for each category select top nterm/ (k**2)
    k_terms = int(len(vsm_matrix.columns)/(len(mi_matrix.index) ** 2))

    selected_terms = set()
    # selected_terms = {row.nlargest(k_terms).index.tolist() for _, row in mi_matrix.iterrows()}
    for _, row in mi_matrix.iterrows():
        selected_terms.update(row.nlargest(k_terms).index.tolist())

    # only keep selected_terms, drop other columns
    vsm_matrix = vsm_matrix[selected_terms]

    print(f'Features: {len(selected_terms)}')

    return vsm_matrix


def _apply_normalization(vsm_matrix):
    row_magnitudes = np.sqrt(
        np.square(vsm_matrix).sum(axis=1)
    )
    vsm_matrix = vsm_matrix.div(row_magnitudes, axis=0)

    return vsm_matrix
