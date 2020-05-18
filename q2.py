"""
This module is for user interface
to evaluate clustering Q2


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

from time import time

from ir3_clustering import evaluate_clustering
from ir3_filing import read_documents, load_object, store_object
from ir3_vsm_indexer import create_vsm_matrix


def main():

    # PARAMETERS

    root_path = r'bbcsport'  # local path to the bbcsport folder (project folder is assumed)
    index_path = r'vsm_index'  # file name for storing VSM index object

    k = 5  # number of clusters to create

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

        #  Apply both feature selection (Mutual Information) and scaling
        vsm_index = create_vsm_matrix(
            documents, selection=True, scaling=True)

        store_object(vsm_index, index_path)

    vsm_matrix, response_vector = vsm_index

    cc = time()

    purity = evaluate_clustering(
        vsm_matrix, k, store_results=store_results)
    dd = time()

    print(f'VSM {index_proc} in : {cc - aa} seconds\n'
          f'Evaluated in : {dd - cc} seconds\n')

    print(f'Purity: {round(purity * 100, 2)} %')


if __name__ == '__main__':
    main()
