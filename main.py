from classes import Corpus, Document, QueriedBM25Corpus, QueriedDocument
from utils import precision_at_k, recall_at_k, average_precision, reciprocal_rank
import streamlit as st


BM25_K = 1.5
BM25_B = 0.75
NUM_DOCUMENTS_OF_INTEREST = 5

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "relevance_feedback" not in st.session_state:
    st.session_state.relevance_feedback = {}

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
            corpus = Corpus(k=BM25_K, b=BM25_B)

            # Store corpus in session state
            st.session_state.corpus = corpus

            # Update progress bar
            progress_bar.progress(10, text="Leyendo documentos...")

            # Add documents to the corpus
            corpus.add_docs_from_directory("docs/")

            # Setup texts for the documents
            corpus.setup_document_texts()

            # Update progress bar
            progress_bar.progress(30, text="Procesando documentos...")

            # Setup tokens for the documents
            corpus.setup_document_tokens()

            # Update progress bar
            progress_bar.progress(45, text="Escribiendo documentos...")

            # Write docs to directory
            corpus.write_docs_to_directory("output/")

            # Update progress bar
            progress_bar.progress(60, text="Calculando matriz de frecuencia...")

            # Build the term frequency matrix
            bm25_matrix = corpus.get_bm25_matrix()

            # Save to session state
            st.session_state.bm25_matrix = bm25_matrix

            # Update progress bar
            progress_bar.progress(75, text="Calculando clusteres...")

            # Cluster documents
            corpus.cluster_documents()

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
    st.markdown("### Buscar:")
    query_text = st.text_input(
        label="Buscar:",
        placeholder="Ingrese un término de búsqueda...",
        label_visibility="hidden",
    )

    if not query_text:
        st.write("Por favor, ingrese un término de búsqueda.")
        return

    with st.status("Procesando consulta...", expanded=True) as status:
        # Initialize progress bar
        progress_bar = st.progress(0, text="Consultando corpus...")

        # Get corpus from session state
        corpus = st.session_state.corpus

        # Get the cluster for the query
        cluster_id = corpus.predict_cluster_for_query(query_text)

        # Query the corpus
        queried_corpus = QueriedBM25Corpus(corpus, query_text)

        # Save to session state
        st.session_state.queried_corpus = queried_corpus

        # Update progress bar
        progress_bar.progress(20, text="Organizando documentos...")

        # Get the documents sorted by score
        documents_by_score: list[QueriedDocument] = (
            queried_corpus.get_documents_by_score()
        )

        # Update progress bar
        progress_bar.progress(60, text="Definiendo pares de relevancia...")
        for document in documents_by_score:
            st.session_state.relevance_feedback[document.get_document().get_id()] = [
                False
            ] * NUM_DOCUMENTS_OF_INTEREST

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
        for i, queried_document in enumerate(documents_by_score):
            with st.container(border=True):
                # Get the document
                document: Document = queried_document.get_document()

                # Name of the document
                st.write(f"**{document.get_name()}**")

                # Snippet of the document
                st.write(f"{snippets[i]}")

                # Select button to view more details
                st.button(
                    "Ver",
                    key=f"go_to_{document.get_id()}",
                    on_click=go_to_document_info,
                    args=(document,),
                )
    else:
        st.write("No se encontraron resultados.")


# Phase 4: Document Info
def document_info():
    st.button("← Volver", on_click=lambda: st.session_state.update(step=2))

    # Get the necessary objects from session state
    selected_document: Document = st.session_state.selected_document
    queried_corpus: QueriedBM25Corpus = st.session_state.queried_corpus

    # Document Info
    st.subheader("Información del Documento")

    with st.container(border=True):
        st.markdown("#### Nombre")
        st.write(selected_document.get_name())

        st.markdown("#### Ruta")
        st.write(selected_document.get_path())

        st.markdown("#### Texto")
        with st.container(border=True):
            st.write(f"{selected_document.get_text()[:400]}...")

        st.divider()

        st.download_button(
            "Descargar Documento",
            data=selected_document.get_file(),
            file_name=selected_document.get_name() + ".pdf",
            mime="application/pdf",
        )

    # Get similarities for current document
    doc_index = queried_corpus.get_corpus().get_document_index_by_id(
        selected_document.get_id()
    )
    similar_docs = queried_corpus.get_similar_documents(
        st.session_state.bm25_matrix[doc_index]
    )[1 : NUM_DOCUMENTS_OF_INTEREST + 1]

    # Similar documents
    st.subheader("Documentos Similares")

    relevant_docs = [None] * NUM_DOCUMENTS_OF_INTEREST

    for i, document in enumerate(similar_docs):
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])

            # Show the document
            with col1:
                st.write(f"**{document.get_name()}**")
                st.button(
                    "Ver",
                    key=f"go_to_{document.get_id()}",
                    on_click=go_to_document_info,
                    args=(document,),
                )

            # Check for relevance
            with col2:
                is_relevant = st.checkbox(
                    "Es relevante?",
                    value=st.session_state.relevance_feedback[
                        selected_document.get_id()
                    ][i],
                    key=f"relevance_{document.get_id()}",
                    on_change=update_relevance_feedback,
                    args=(
                        selected_document.get_id(),
                        i,
                    ),
                )
                if is_relevant:
                    relevant_docs[i] = document
                else:
                    relevant_docs[i] = None

    # Metric Evaluation
    st.subheader("Evaluación de Métricas")

    filtered_docs = list(filter(lambda x: x is not None, relevant_docs))

    precision = precision_at_k(
        filtered_docs,
        similar_docs,
        NUM_DOCUMENTS_OF_INTEREST,
    )

    recall = recall_at_k(
        filtered_docs,
        similar_docs,
        NUM_DOCUMENTS_OF_INTEREST,
    )

    average_precision_value = round(
        average_precision(
            filtered_docs,
            similar_docs,
        ),
        2,
    )

    reciprocal_rank_value = reciprocal_rank(
        filtered_docs,
        similar_docs,
    )

    a, b = st.columns(2)
    c, d = st.columns(2)
    e, f = st.columns(2)

    a.metric("Recomendados", len(similar_docs), border=True)
    b.metric("Relevantes", len(filtered_docs), border=True)

    c.metric("Precisión@k", precision, border=True)
    d.metric("Recall@k", recall, border=True)

    e.metric("Precisión Media", average_precision_value, border=True)
    f.metric("Rango Reciproco", reciprocal_rank_value, border=True)


def update_relevance_feedback(document_id, index):
    st.session_state.relevance_feedback[document_id][index] = (
        not st.session_state.relevance_feedback[document_id][index]
    )


def go_to_document_info(document: Document):
    st.session_state.selected_document = document
    st.session_state.step = 3


if st.session_state.step == 1:
    initial_config()

if st.session_state.step == 2:
    query_system()

if st.session_state.step == 3:
    document_info()
