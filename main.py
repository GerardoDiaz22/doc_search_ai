from classes import Corpus, QueriedBM25Corpus, QueriedDocument
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
                progress_bar.progress(25, text="Cargando documentos...")

                # Add documents to the corpus
                corpus.add_docs_from_directory("docs/")

                # Update progress bar
                progress_bar.progress(75, text="Escribiendo documentos...")

                # Write docs to directory
                corpus.write_docs_to_directory("output/")

                # HERE WOULD BE THE CLUSTERING PHASE
                # REMEMBER TO UPDATE THE PROGRESS BAR

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

    # Query the corpus
    queried_corpus = QueriedBM25Corpus(corpus, query_text, k=1.5, b=0.75)

    # Build the term frequency matrix
    tf_matrix = queried_corpus.get_tf_matrix()

    # Phase 3: Displaying Results

    # Get the documents sorted by score
    documents_by_score: list[QueriedDocument] = queried_corpus.get_documents_by_score()

    # Display the results
    st.write(f"Se han encontrado {len(documents_by_score)} resultados:")

    if documents_by_score:
        for document in documents_by_score:
            similar_docs = queried_corpus.get_similar_documents(
                tf_matrix[
                    queried_corpus.get_corpus().get_document_index_by_id(
                        document.get_document().get_id()
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
                    f"{queried_corpus.get_document_snippet_by_id(document.get_document().get_id())}"
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
