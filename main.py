from classes import Corpus, QueriedBM25Corpus, QueriedDocument
import streamlit as st

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1

st.title("Buscador de Documentos")


# Phase 1: Initialization
def initial_config():
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
            progress_bar.progress(25, text="Leyendo documentos...")

            # Add documents to the corpus
            corpus.add_docs_from_directory("docs/")

            # Setup texts for the documents
            corpus.setup_document_texts()

            # Update progress bar
            progress_bar.progress(50, text="Procesando documentos...")

            # Setup tokens for the documents
            corpus.setup_document_tokens()

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

        # Move to the next step
        st.session_state.step = 2
        st.rerun()


def query_system():
    st.button("← Volver", on_click=lambda: st.session_state.update(step=1))

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

        # Get corpus from session state
        corpus = st.session_state.corpus

        # Query the corpus
        queried_corpus = QueriedBM25Corpus(corpus, query_text, k=1.5, b=0.75)

        # Save to session state
        st.session_state.queried_corpus = queried_corpus

        # Update progress bar
        progress_bar.progress(20, text="Organizando documentos...")

        # Get the documents sorted by score
        documents_by_score: list[QueriedDocument] = (
            queried_corpus.get_documents_by_score()
        )

        # Update progress bar
        progress_bar.progress(40, text="Calculando matriz de frecuencia...")

        # Build the term frequency matrix
        tf_matrix = queried_corpus.get_tf_matrix()

        # Save to session state
        st.session_state.tf_matrix = tf_matrix

        # Update progress bar
        progress_bar.progress(80, text="Generando snippets...")

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

                # Select button to view more details
                st.button(
                    "Ver",
                    key=f"view_{i}",
                    on_click=lambda: st.session_state.update(
                        selected_document=document, step=3
                    ),
                )

                st.markdown("---")
    else:
        st.write("No se encontraron resultados.")


# Phase 4: Document Info
def document_info():
    st.button("← Volver", on_click=lambda: st.session_state.update(step=2))

    st.write("**Información del documento**")

    # Get the necessary objects from session state
    selected_document: QueriedDocument = st.session_state.selected_document
    queried_corpus: QueriedBM25Corpus = st.session_state.queried_corpus

    # Get similarities for current document
    doc_index = queried_corpus.get_corpus().get_document_index_by_id(
        selected_document.get_document().get_id()
    )
    similar_docs = queried_corpus.get_similar_documents(
        st.session_state.tf_matrix[doc_index]
    )

    # Document Info
    st.write(f"**Nombre:** {selected_document.get_document().get_name()}")

    # Similar documents
    st.write(
        f"**Similares:** {
        ', '.join([doc.get_name() for doc in similar_docs][1:5])
    }"
    )


if st.session_state.step == 1:
    initial_config()

if st.session_state.step == 2:
    query_system()

if st.session_state.step == 3:
    document_info()
