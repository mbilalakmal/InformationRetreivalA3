"""
This module defines pre-processing methods for
text documents. The steps are mentioned below:
1. Tokenize
2. Case-fold
3. Remove stopwords
3. Stem


(C) 2020 Muhammad Bilal Akmal, 17K-3669
"""

import re

from nltk.stem import PorterStemmer


def extract_features(document: str):
    tokens = _get_tokens(document)

    tokens = [token.lower() for token in tokens]

    tokens = _remove_stopwords(tokens)

    stemmer = PorterStemmer()
    terms = [stemmer.stem(token) for token in tokens]

    return terms


def _get_tokens(document: str):
    pattern = r'\W+'
    tokens = re.split(pattern, document)
    return tokens


def _remove_stopwords(tokens: list):
    filename = r'stopwords.txt'
    with open(filename, 'r') as txt_file:
        stopwords = set(txt_file.read().split())

    tokens = [
        token for token in tokens if (token not in stopwords and len(token) > 1)
    ]
    return tokens
