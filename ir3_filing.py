"""
This module defines filing methods for
storing, loading, and traversing files and folders.


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

import pickle
from os import walk

from ir3_document import Document


def read_documents(root_path: str):
    """
    Reads documents from `root_path` and returns
    a dictionary of documents.
    Folders directly below `root_path` are considered
    the category labels (classes).
    All document files are assumed to be found inside
    these subfolders.

    :param root_path: Folder directory of corpus
    :return: Dictionary containing the documents
    """
    category_folders = next(walk(root_path, '.'))[1]
    documents = {}  # dictionary containing the documents

    for cat_folder in category_folders:
        # get filenames from current folder
        folder_path = fr'{root_path}\{cat_folder}'
        doc_filenames = next(walk(folder_path, '.'))[2]

        for doc_filename in doc_filenames:
            # open and read each file
            file_path = folder_path + fr'\{doc_filename}'

            with open(file_path, mode='r') as doc_file:
                body = doc_file.read()

            # create and save Document object
            doc_id = f'{cat_folder}_{doc_filename[:doc_filename.rfind(".")]}'
            document = Document(
                doc_id=doc_id, path=file_path, body=body, category=cat_folder
            )
            documents[doc_id] = document

    return documents


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        return None


def store_object(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
