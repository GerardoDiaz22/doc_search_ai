import pymupdf
import spacy
import os
import numpy as np
import streamlit as st
from collections import defaultdict


# Load spaCy Spanish model
nlp = spacy.load("es_core_news_sm")


# Increase the limit to 2,000,000 characters
nlp.max_length = 2000000 


def build_term_frequency_matrix(corpus_tokens):

    #Build a matrix representing the table of word values ​​in cosine metrics

    # get all unique terms in the corpus
    all_terms = set(term for doc_tokens in corpus_tokens for term in doc_tokens)

    all_terms = list(all_terms)


    # Create a term-frequency matrix for each document in the corpus
    num_docs = len(corpus_tokens)
    num_terms = len(all_terms)

    tf_matrix = np.zeros((num_docs, num_terms), dtype=int)

    # Fill the matrix with the frequencies of the terms
    term_to_index = {term: i for i, term in enumerate(all_terms)}

    for doc_idx, doc_tokens in enumerate(corpus_tokens):
        term_counts = defaultdict(int)
        for term in doc_tokens:
            term_counts[term] += 1
        for term, count in term_counts.items():
            tf_matrix[doc_idx, term_to_index[term]] = count

    return tf_matrix


#makes use of the cosine metric formula
def cosine_similarity(matrix, query_vector):

    dot_product = np.dot(matrix, query_vector)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)

   # Calculate cosine similarity
    similarities = dot_product / (matrix_norms * query_norm)
    return similarities

def get_cosine_similarities(selected_doc_index, tf_matrix):

    # Get the vector of the selected document
    selected_doc_vector = tf_matrix[selected_doc_index]

    similarities = cosine_similarity(tf_matrix, selected_doc_vector)
    return similarities


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


def write_text_to_file(file_path, text):
    # Create a file to write the text to
    out = open(file_path, "wb")

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
    doc_ranking = []
    avg_doc_length = get_avg_doc_length(corpus)
    for doc in corpus:
        doc_score = 0
        highest_term_score = 0
        best_term = None
        for term in query:
            tf_score = tf(term, doc)
            idf_score = idf(term, corpus)
            theta = len(doc) / avg_doc_length
            term_score = (
                idf_score * (tf_score * (k + 1)) / (tf_score + k * (1 - b + b * theta))
            )
            doc_score += term_score
            if term_score > highest_term_score:
                highest_term_score = term_score
                best_term = term
        doc_ranking.append({"score": doc_score, "best_term": best_term})
    return doc_ranking


def sorted_indices_by_value(vector):
    # Sort the vector based on values in descending order
    sorted_vector = sorted(vector, key=lambda x: x["score"], reverse=True)

    sorted_vector_with_indices = []
    for item in sorted_vector:
        item["index"] = vector.index(item)
        sorted_vector_with_indices.append(item)

    return sorted_vector_with_indices


def handle_corpus_from_docs(docs_path, output_path):
    corpus_doc_paths = []
    corpus_tokens = []
    for file_name in os.listdir(docs_path):
        # Construct the full file path
        full_path = os.path.join(docs_path, file_name)

        # Check if it's a pdf file
        if file_name.endswith(".pdf"):
            # STEP 1.1: Read the PDF file
            pdf_text = get_text_from_pdf(full_path)

            # STEP 1.2: Remove special characters, convert to lowercase, remove stopwords, tokenize and lemmatize
            pdf_tokens = tokenize_clean_text(pdf_text)

            # STEP 1.3: Write the cleaned text to a file
            output_file_path = os.path.join(
                output_path, file_name.replace(".pdf", ".txt")
            )
            clean_text = " ".join(pdf_tokens)
            write_text_to_file(output_file_path, clean_text)

            # STEP 2.1: Append the tokens to the corpus
            corpus_tokens.append(pdf_tokens)

            # STEP 2.2: Save the path to the document
            corpus_doc_paths.append(output_file_path)

    return corpus_doc_paths, corpus_tokens


def snippets_from_docs(doc_info):
    snippets = []
    for doc_path, best_token in doc_info:
        # Read the text from the document
        text = read_text_from_file(doc_path)

        # Tokenize the text
        doc_tokens = tokenize_clean_text(text)

        # Find the index of the first occurrence of the query in the document
        try:
            query_index = doc_tokens.index(best_token)
        except ValueError:
            query_index = 0

        # Get the snippet
        snippet = doc_tokens[query_index : query_index + 5]

        # Append the snippet to the list
        snippets.append(" ".join(snippet))

    return snippets


# Main


def main():
    st.title("Buscador de Documentos")

    # STEP 1-2: Read the documents, handle them and create the corpus
    # NOTE: This could be break down into two or more functions for readability, but for better performance i'll keep it as is
    corpus_doc_paths, corpus_tokens = handle_corpus_from_docs("docs", "output")

     # STEP 2.1: Build the term-frequency matrix
    tf_matrix = build_term_frequency_matrix(corpus_tokens)

    # STEP 3: Read the query from the user
    query = st.text_input("Buscar:", placeholder="Ingrese un término de búsqueda...")
    tokenized_query = tokenize_clean_text(query)

    # STEP 4: Rank the documents with bm25
    if tokenized_query:
        k = 1.5
        b = 0.75
        doc_ranking = get_bm25_score(tokenized_query, corpus_tokens, k, b)
        sorted_docs = sorted_indices_by_value(doc_ranking)

        # STEP 5: Display the results
        doc_info = [
            (corpus_doc_paths[doc["index"]], doc["best_term"]) for doc in sorted_docs
        ]

        # NOTE: This also could be done above, but lets just do it here so it's easier to understand the flow of the algorithm
        text_snippets = snippets_from_docs(doc_info)

        results = []
        for (doc_path, _), snippet in zip(doc_info, text_snippets):
            results.append({"doc_path": doc_path, "snippet": snippet})

        st.write(f"Se han encontrado {len(results)} resultados:")

        if results:
            for result in results:
                st.write(f"Documento: {result['doc_path']}")
                st.write(f"{result['snippet']}")

                # STEP 6: Recommend similar documents
                st.write("##### Documentos Similares")
                 #  Get index of the current document
                current_doc_index = corpus_doc_paths.index(result['doc_path'])

                #Calculate similarities based on the current document
                similarities = get_cosine_similarities(current_doc_index, tf_matrix)
                similar_docs = sorted(
                    zip(corpus_doc_paths, similarities), key=lambda x: x[1], reverse=True
                )

                # Excludes the current document and displays the 3 most similar ones
                for doc_path, similarity in similar_docs[1:4]:
                    st.write(
                        f"Documento: {doc_path} | Similitud del coseno: {similarity:.3f}"
                    )
                st.write("----")

        else:
            st.write("No se encontraron resultados.")
    else:
        st.write("Ingrese un término de búsqueda para comenzar.")


if __name__ == "__main__":
    main()
