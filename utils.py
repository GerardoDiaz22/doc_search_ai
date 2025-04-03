import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load spaCy Spanish model
nlp = spacy.load(
    "es_core_news_sm", disable=["parser", "ner", "attribute_ruler", "tok2vec"]
)

# Configuration the nlp model
nlp.max_length = 2000000


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


def precision_at_k(relevant_docs, retrieved_docs, k):
    top_k = retrieved_docs[:k]
    relevant_in_top = len(set(top_k) & set(relevant_docs))
    return relevant_in_top / k


def recall_at_k(relevant_docs, retrieved_docs, k):
    top_k = retrieved_docs[:k]
    relevant_in_top = len(set(top_k) & set(relevant_docs))
    return relevant_in_top / len(relevant_docs) if relevant_docs else 0


def average_precision(relevant_docs, retrieved_docs):
    relevant_indices = [
        i + 1 for i, doc in enumerate(retrieved_docs) if doc in relevant_docs
    ]
    return (
        sum(precision_at_k(relevant_docs, retrieved_docs, i) for i in relevant_indices)
        / len(relevant_docs)
        if relevant_docs
        else 0
    )


def reciprocal_rank(relevant_docs, retrieved_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0


def get_optimal_k_clusters(matrix):
    silhouette_scores = []
    for k in range(2, 16):
        k_means_model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = k_means_model.fit_predict(matrix)
        silhouette_scores.append((k, silhouette_score(matrix, clusters)))
    return max(silhouette_scores, key=lambda x: x[1])[0]
