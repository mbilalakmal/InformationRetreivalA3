from time import time

from ir3_classification import evaluate_classification
from ir3_filing import read_documents
from ir3_vsm_indexer import create_vsm_matrix


def main():
    root_path = root_path = r'G:\My Drive\University\Spring (2020)\Information Retrieval\Assignment3\bbcsport'

    aa = time()

    documents = read_documents(root_path)
    bb = time()

    vsm_matrix, response_vector = create_vsm_matrix(documents)
    cc = time()

    k = 3
    ratio = 0.7

    accuracy = evaluate_classification(vsm_matrix, response_vector, k, ratio)
    dd = time()

    print(f'Read in : {bb - aa} seconds\n'
          f'VSM created in : {cc - bb} seconds\n'
          f'Evaluated in : {dd - cc} seconds\n')

    print(accuracy * 100)


if __name__ == '__main__':
    main()
