"""
This module is for user interface
to evaluate classification Q1


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

from time import time

from ir3_classification import evaluate_classification
from ir3_filing import read_documents, load_object, store_object
from ir3_vsm_indexer import create_vsm_matrix


def main():

    # PARAMETERS

    root_path = r'bbcsport'  # local path to the bbcsport folder (project folder is assumed)
    index_path = r'vsm_index'  # file name for storing VSM index object

    k = 3  # number of neighbors for kNN
    ratio = 0.7  # ratio of training data

    store_results = True  # True means a q1_results.csv will be stored

    fresh_option = True  # True means vsm_index will be created and stored from scratch
    # False means if vsm_index file is found it will be loaded

    aa = time()

    vsm_index = load_object(index_path)

    # If index is not stored, create it and store
    index_proc = 'Loaded'
    if vsm_index is None or fresh_option is True:
        index_proc = 'Created and Stored'
        documents = read_documents(root_path)

        # Don't apply feature selection but apply scaling (TFIDF)
        vsm_index = create_vsm_matrix(
            documents, selection=False, scaling=True)

        store_object(vsm_index, index_path)

    vsm_matrix, response_vector = vsm_index

    cc = time()

    accuracy = evaluate_classification(
        vsm_matrix, response_vector, k, ratio, store_results=store_results)
    dd = time()

    print(f'VSM {index_proc} in : {cc - aa} seconds\n'
          f'Evaluated in : {dd - cc} seconds\n')

    print(f'Accuracy: {round(accuracy * 100, 2)} %')


if __name__ == '__main__':
    main()
