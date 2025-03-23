from classes import Corpus, QueriedBM25Corpus, QueriedDocument
import streamlit as st

# TODO: change all the by_name handlings to use by_index or by_id
# TODO: change the QueriedDocument to save the (token, score) pair


def main():
    st.title("Buscador de Documentos")

    # Initialize the corpus
    corpus = Corpus()
    corpus.add_docs_from_directory("docs/")

    # Write docs to directory
    corpus.write_docs_to_directory("output/")

    # Read the query from the user
    query_text = st.text_input(
        "Buscar:", placeholder="Ingrese un término de búsqueda..."
    )

    if not query_text:
        st.write("Por favor, ingrese un término de búsqueda.")
        return

    # Query the corpus
    queried_corpus = QueriedBM25Corpus(corpus, query_text, k=1.5, b=0.75)

    # Build the term frequency matrix
    tf_matrix = queried_corpus.get_tf_matrix()

    # Get the documents sorted by score
    documents_by_score: list[QueriedDocument] = queried_corpus.get_documents_by_score()

    # Display the results
    st.write(f"Se han encontrado {len(documents_by_score)} resultados:")

    if documents_by_score:
        for document in documents_by_score:
            similar_docs = queried_corpus.get_similar_documents(
                tf_matrix[
                    queried_corpus.get_corpus().get_index_by_name(
                        document.get_document().get_name()
                    )
                ]
            )
            with st.container():
                # Name of the document
                st.write(f"**{document.get_document().get_name()}**")
                # File path of the document
                st.write(f"{document.get_document().get_path()}")
                # Snippet of the document
                st.write(
                    f"{queried_corpus.get_snippet_from_doc_by_score(document.get_document().get_name())}"
                )

                # Similar documents
                st.write(
                    f"Similares: {
                    ', '.join([doc.get_name() for doc in similar_docs][1:])
                }"
                )
                st.markdown("---")
    else:
        st.write("No se encontraron resultados.")


if __name__ == "__main__":
    main()
