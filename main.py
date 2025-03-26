import pymupdf
import os
import numpy as np
import uuid
import shutil
from utils import tokenize_and_clean_text, bm25_score
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Optional
from classes import *

# This is main.py, modified to integrate clustering.
import streamlit as st


def main():
    st.title("Buscador de Documentos")

    # Initialize session state
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False

    # Phase 1: Initialization

    if not st.session_state.setup_complete:
        st.write("**Configuración del sistema**")
        st.write(
            "Por favor, haga clic en el botón para iniciar la configuración del sistema. Puede tardar unos minutos dependiendo de la cantidad de documentos que tenga."
        )
        if st.button("Iniciar Configuración"):
            # Create a status container and progress bar
            with st.status(
                "Configurando sistema, por favor espere...", expanded=True
            ) as status:
                # Initialize progress bar
                progress_bar = st.progress(0, text="Inicializando corpus...")

                # Initialize the corpus
                corpus = Corpus()

                # Store corpus in session state
                st.session_state.corpus = corpus

                # Update progress bar
                progress_bar.progress(20, text="Leyendo documentos...")

                # Add documents to the corpus
                corpus.add_docs_from_directory("docs/")

                # Setup texts for the documents
                corpus.setup_document_texts()

                # Update progress bar
                progress_bar.progress(40, text="Procesando documentos...")

                # Setup tokens for the documents
                corpus.setup_document_tokens()

                # Update progress bar
                progress_bar.progress(60, text="Clustering documentos...") # UPDATED PROGRESS BAR

                # CLUSTERING PHASE
                corpus.cluster_documents()  # You can adjust the number of clusters
                st.session_state.corpus = corpus # Update the session state after clustering.

                # Update progress bar
                progress_bar.progress(80, text="Escribiendo documentos...")

                # Write docs to directory
                corpus.write_docs_to_directory("output/")

                # Complete setup
                progress_bar.progress(100, text="Configuración completada!")
                status.update(
                    label="Configuración completada!", state="complete", expanded=False
                )

            st.session_state.setup_complete = True
            st.rerun()
        return

    # Get corpus from session state
    corpus = st.session_state.corpus

    # Phase 2: Querying

    # Read the query from the user
    query_text = st.text_input(
        "Buscar:", placeholder="Ingrese un término de búsqueda..."
    )

    if not query_text:
        st.write("Por favor, ingrese un término de búsqueda.")
        return

    with st.status("Procesando consulta...", expanded=True) as status:
        # Initialize progress bar
        progress_bar = st.progress(0, text="Consultando corpus...")

        # Predict the cluster for the query
        predicted_cluster_id = corpus.predict_cluster_for_query(query_text)

        # Update progress bar
        progress_bar.progress(10, text="Prediciendo cluster...")

        if predicted_cluster_id is not None:
            st.write(f"Buscando en el clúster: {predicted_cluster_id}")
        else:
            st.write("No se pudo predecir el clúster. Buscando en todos los documentos.")

        # Query the corpus, now taking into account the predicted cluster.
        queried_corpus = QueriedBM25Corpus(corpus, query_text, k=1.5, b=0.75, predicted_cluster_id=predicted_cluster_id) #PASS THE CLUSTER ID

        # Update progress bar
        progress_bar.progress(30, text="Organizando documentos...")

        # Get the documents sorted by score
        documents_by_score: list[QueriedDocument] = (
            queried_corpus.get_documents_by_score()
        )

        # Update progress bar
        progress_bar.progress(50, text="Calculando matriz de frecuencia...")

        # Build the term frequency matrix
        tf_matrix = queried_corpus.get_tf_matrix()

        # Update progress bar
        progress_bar.progress(70, text="Buscando documentos similares...")

        # Get similarities for each document
        similar_docs_list: list[list[Document]] = []
        if tf_matrix is not None: #Check tf_matrix
            for document in documents_by_score:
                doc_id = document.get_document().get_id()
                doc_index = queried_corpus.get_corpus().get_document_index_by_id(doc_id)
                if doc_index is not None and doc_index < len(tf_matrix):  # Ensure index is valid
                        query_vector = tf_matrix[doc_index]
                        similar_docs = queried_corpus.get_similar_documents(query_vector)
                        if similar_docs is not None:
                            similar_docs_list.append(similar_docs)
                        else:
                            similar_docs_list.append([]) # Append an empty list if similar_docs is None
                else:
                    similar_docs_list.append([])  # Handle missing document index
        else:
           st.error("Failed to compute TF-IDF matrix. Cannot retrieve similar documents.")

        # Update progress bar
        progress_bar.progress(90, text="Generando snippets...")

        # Generate snippets for the documents
        snippets = []
        for document in documents_by_score:
            snippet = queried_corpus.get_document_snippet_by_id(
                document.get_document().get_id()
            )
            snippets.append(snippet)

        # Complete querying
        progress_bar.progress(100, text="Consulta completada!")
        status.update(label="Consulta completada!", state="complete", expanded=False)

    # Phase 3: Displaying Results

    # Display the results
    st.write(f"Se han encontrado {len(documents_by_score)} resultados:")

    if documents_by_score:
        for i, document in enumerate(documents_by_score):
            with st.container():
                # Name of the document
                st.write(f"**{document.get_document().get_name()}**")
                # File path of the document
                st.write(f"{document.get_document().get_path()}")
                # Snippet of the document
                st.write(f"{snippets[i]}")

                # Similar documents
                if i < len(similar_docs_list):  # Make sure index is valid
                    st.write(
                        f"Similares: {', '.join([doc.get_name() for doc in similar_docs_list[i][1:5]])}"
                    )
                else:
                    st.write("No similar documents found.")

                st.markdown("---")
    else:
        st.write("No se encontraron resultados.")


if __name__ == "__main__":
    main()