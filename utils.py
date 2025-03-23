import spacy
import numpy as np

# Load spaCy Spanish model
nlp = spacy.load("es_core_news_sm")


def tokenize_and_clean_text(text: str) -> list[str]:
    # Convert to lowercase
    text = text.lower()

    # Tokenization with spaCy
    spacy_doc = nlp(text)

    # Stopword removal and lemmatization
    tokens = [
        token.lemma_
        for token in spacy_doc
        if not token.is_stop and (token.is_alpha or token.is_digit)
    ]

    # Return the tokens
    return tokens


def tf(term: str, document: list[str]) -> float:
    return document.count(term)


def idf(term: str, documents: list[list[str]]) -> float:
    N = len(documents)
    df = sum([1 for doc in documents if term in doc])
    if df == 0:
        return 0
    else:
        return np.log(N / df)


def bm25_score(
    term: str,
    document: list[str],
    documents: list[list[str]],
    avg_doc_length: float,
    k: float,
    b: float,
) -> float:
    tf_score = tf(term, document)
    idf_score = idf(term, documents)
    theta = len(document) / avg_doc_length
    return idf_score * (tf_score * (k + 1)) / (tf_score + k * (1 - b + b * theta))
