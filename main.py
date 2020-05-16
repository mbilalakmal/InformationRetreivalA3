from time import time

from ir3_classification import evaluate_classification
from ir3_filing import read_documents, load_object, store_object
from ir3_vsm_indexer import create_vsm_matrix


def main():
    root_path = r'G:\My Drive\University\Spring (2020)\Information Retrieval\Assignment3\bbcsport'
    index_path = r'vsm_index'

    aa = time()

    vsm_index = load_object(index_path)

    # If index is not stored, create it and store
    if vsm_index is None:
        documents = read_documents(root_path)

        vsm_index = create_vsm_matrix(documents)

        store_object(vsm_index, index_path)

    vsm_matrix, response_vector = vsm_index

    cc = time()

    k = 3
    ratio = 0.7

    accuracy = evaluate_classification(vsm_matrix, response_vector, k, ratio)
    dd = time()

    print(f'VSM created in : {cc - aa} seconds\n'
          f'Evaluated in : {dd - cc} seconds\n')

    print(accuracy * 100)


if __name__ == '__main__':
    main()
