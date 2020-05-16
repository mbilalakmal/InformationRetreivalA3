"""
This module defines a Document object
which contains the features and the label
of an actual text document.


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

from ir3_preprocessing import extract_features


class Document:
    def __init__(
            self,
            doc_id: str,
            path: str,
            body: str,
            category: str
    ):
        self.doc_id = doc_id
        self.path = path
        self.body = body
        self.category = category
        self.terms = extract_features(body)

    def __repr__(self):
        return (
            f'Doc ID: {self.doc_id}\n'
            f'Path: {self.path}\n'
            # f'Title: {self.title}\n'
            f'Body: {self.body[:30]}\n'
            f'Category: {self.category}\n'
        )
