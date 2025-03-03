import pymupdf
import spacy
import os
from rank_bm25 import BM25Okapi
import numpy as np


# Load spaCy Spanish model
nlp = spacy.load("es_core_news_sm")


def get_text_from_pdf(pdf_path):
    # Open the PDF file
    pdf_doc = pymupdf.open(pdf_path)

    # Initialize an empty string to store the intermediate text
    pdf_text = ""

    # Iterate over all pages in the document
    for page in pdf_doc:
        # Get plain text of the page
        text = page.get_text()

        # Append the text to the intermediate string
        pdf_text += text

        # Write the page break character
        # TODO: Maybe adding this character is not necessary for the model
        pdf_text += chr(12)

    # Close the PDF file
    pdf_doc.close()

    return pdf_text


def tokenize_clean_text(text):
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


def write_text_to_file(file_name, text):
    # Create a file to write the text to
    out = open(os.path.join(OUTPUT_PATH, file_name), "wb")

    # Write the text to the output file
    out.write(text.encode("utf8"))

    # Close the output file
    out.close()


def read_text_from_file(file_name):
    # Open the file
    with open(file_name, "r", encoding="utf-8") as file:
        # Read the text
        text = file.read()

    return text


def tf(term, doc):
    return doc.count(term)


def idf(term, corpus):
    N = len(corpus)
    df = sum([1 for doc in corpus if term in doc])
    if df == 0:
        return 0
    else:
        return np.log(N / df)


def get_tf_idf_score(query, corpus):
    scores = []
    for doc in corpus:
        score = 0
        for term in query:
            tf_score = tf(term, doc)
            idf_score = idf(term, corpus)
            score += tf_score * idf_score
        scores.append(score)
    return scores


def get_avg_doc_length(corpus):
    return sum([len(doc) for doc in corpus]) / len(corpus)


def get_bm25_score(query, corpus, k, b):
    scores = []
    avg_doc_length = get_avg_doc_length(corpus)
    for doc in corpus:
        score = 0
        for term in query:
            tf_score = tf(term, doc)
            idf_score = idf(term, corpus)
            theta = len(doc) / avg_doc_length
            score += (
                idf_score * (tf_score * (k + 1)) / (tf_score + k * (1 - b + b * theta))
            )
        scores.append(score)
    return scores


def sorted_indices_by_value(vector):
    # Create a list of tuples (value, index)
    indexed_vector = list(enumerate(vector))

    # Sort the vector based on values in descending order
    sorted_vector = sorted(indexed_vector, key=lambda x: x[1], reverse=True)

    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_vector]

    return sorted_indices


# Main

DOCS_PATH = "docs"
OUTPUT_PATH = "output"

corpus_doc_paths = []
corpus_tokens = []

for file_name in os.listdir(DOCS_PATH):
    # Construct the full file path
    full_path = os.path.join(DOCS_PATH, file_name)

    # Check if it's a pdf file
    if file_name.endswith(".pdf"):
        # STEP 1.1: Read the PDF file
        pdf_text = get_text_from_pdf(full_path)

        # STEP 1.2: Remove special characters, convert to lowercase, remove stopwords, tokenize and lemmatize
        pdf_tokens = tokenize_clean_text(pdf_text)

        # STEP 1.3: Write the cleaned text to a file
        output_file_name = file_name.replace(".pdf", ".txt")
        clean_text = " ".join(pdf_tokens)
        write_text_to_file(output_file_name, clean_text)

        # STEP 2.1: Append the tokens to the corpus
        corpus_tokens.append(pdf_tokens)

        # STEP 2.2: Save the path to the document
        corpus_doc_paths.append(output_file_name)


# STEP 2.3: Create the BM25 model
bm25 = BM25Okapi(corpus_tokens)

# STEP 3: Read the query from the user
query = "mejores juegos rpg y acción"
tokenized_query = tokenize_clean_text(query)

# STEP 4: Rank the documents with bm25
k = 1.5
b = 0.75
doc_scores = get_bm25_score(tokenized_query, corpus_tokens, k, b)
sorted_indices = sorted_indices_by_value(doc_scores)
print(doc_scores, sorted_indices)

for i, doc_index in enumerate(sorted_indices):
    print(f"Rank: {i}, Document: {corpus_doc_paths[doc_index]}")
