import pymupdf
import os
import numpy as np
import uuid
import shutil
from utils import (
    tokenize_and_clean_text,
    bm25_score,
    compute_optimal_k_clusters_silhouette,
    compute_optimal_k_clusters_elbow,
)
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

NUM_COMPONENTS = 5
NUM_NEIGHBORS = 5


class Document:
    def __init__(self, path_to_document):
        if not path_to_document.endswith(".pdf"):
            raise ValueError("The document must be a PDF file")

        self.path_to_document: str = path_to_document
        self.name: str = path_to_document.split("/")[-1].split(".")[0]
        self.id: str = str(uuid.uuid4())
        self.text: str | None = None
        self.tokens: list[str] | None = None
        self.cluster_id: str | None = None

    def get_path(self) -> str:
        return self.path_to_document

    def setup_text(self) -> None:
        try:
            if self.text is None:
                self.text = self._read_text_from_pdf(self.path_to_document)
        except Exception as e:
            print(f"An error occurred: {e}")
            self.text = None

    def get_text(self) -> str | None:
        try:
            if self.text is None:
                self.setup_text()
            return self.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def setup_tokens(self) -> None:
        try:
            if self.tokens is None:
                self.tokens = tokenize_and_clean_text(self.get_text())
        except Exception as e:
            print(f"An error occurred: {e}")
            self.tokens = None

    def get_tokens(self) -> list[str] | None:
        try:
            if self.tokens is None:
                self.setup_tokens()
            return self.tokens
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_name(self) -> str:
        return self.name

    def get_id(self) -> str:
        return self.id

    def write_tokens_to_path(self, path_to_output: str) -> None:
        try:
            with open(path_to_output, "wb") as out:
                out.write(" ".join(self.get_tokens()).encode("utf8"))
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_file(self) -> str:
        with open(self.path_to_document, "rb") as file:
            return file.read()

    def get_cluster_id(self) -> str | None:
        return self.cluster_id

    def set_cluster_id(self, cluster_id: str) -> None:
        self.cluster_id = cluster_id

    @staticmethod
    def _read_text_from_pdf(path_to_pdf: str) -> str | None:
        try:
            # Initialize an empty list to store text from each page
            text_list = []

            # Open the PDF file
            with pymupdf.open(path_to_pdf) as pdf_doc:
                for page in pdf_doc:
                    # Extract text from the page
                    text_list.append(page.get_text())

                    # Write the page break character
                    # TODO: Maybe adding this character is not necessary for the model
                    text_list.append(chr(12))

                # Join the text from all pages into a single string
                pdf_text = "".join(text_list)

            return pdf_text
        except pymupdf.FileNotFoundError:
            print(f"The file '{path_to_pdf}' does not exist")
            return None

    @staticmethod
    def _read_text_from_file(path_to_document: str) -> str | None:
        try:
            with open(path_to_document, "r", encoding="utf-8") as document_file:
                return document_file.read()
        except FileNotFoundError:
            print(f"The file '{path_to_document}' does not exist")
            return None


class Corpus:
    def __init__(self, k: float, b: float):
        self.documents: list[Document] = []
        self.tokens: list[list[str]] | None = None
        self.k: float = k
        self.b: float = b
        self.bm25_matrix = None
        self.reduced_matrix = None
        self.k_means_model = None
        self.pipeline = None
        self.vocabulary: list[str] | None = None
        self.token_to_index: dict[str, int] | None = None
        self.idf_vector: np.ndarray | None = None

    def add_document(self, document: Document) -> None:
        self.documents.append(document)
        self.tokens = None

    def get_documents(self) -> list[Document]:
        return self.documents

    def get_tokens_by_docs(self) -> list[list[str]] | None:
        try:
            if self.tokens is None:
                self.tokens = []
                for document in self.documents:
                    self.tokens.append(document.get_tokens())
            return self.tokens
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def add_docs_from_directory(self, path_to_dir: str) -> list[str]:
        try:
            for file_name in os.listdir(path_to_dir):
                # Construct the full path to the document
                path_to_document = os.path.join(path_to_dir, file_name)

                # Check if it's a pdf file
                if file_name.endswith(".pdf"):
                    # Create a Document object
                    document: Document = Document(path_to_document)

                    # Save the document to the corpus
                    self.add_document(document)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_avg_doc_length(self) -> float:
        return sum([len(doc.get_tokens()) for doc in self.documents]) / len(
            self.documents
        )

    def write_docs_to_directory(self, path_to_dir: str) -> None:
        try:
            # Remove the existing directory and all its contents
            if os.path.exists(path_to_dir):
                shutil.rmtree(path_to_dir)

            # Create a fresh directory
            os.makedirs(path_to_dir)

            for document in self.documents:
                # Construct the full path to the output file
                path_to_output = os.path.join(
                    path_to_dir, document.get_name() + "." + document.get_id() + ".txt"
                )
                # Write the document tokens to the output file
                document.write_tokens_to_path(path_to_output)
        except Exception as e:
            print(f"An error occurred: {e}")

    def setup_document_texts(self) -> None:
        try:
            for document in self.documents:
                document.setup_text()
        except Exception as e:
            print(f"An error occurred: {e}")

    def setup_document_tokens(self) -> None:
        try:
            for document in self.documents:
                document.setup_tokens()
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_document_by_id(self, document_id: str) -> Document | None:
        try:
            for document in self.documents:
                if document.get_id() == document_id:
                    return document
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_document_index_by_id(self, document_id: str) -> int | None:
        try:
            for index, document in enumerate(self.documents):
                if document.get_id() == document_id:
                    return index
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_bm25_matrix(self) -> list[list[float]] | None:
        try:
            if self.bm25_matrix is None:
                corpus_tokens_by_docs: list[list[str]] = self.get_tokens_by_docs()
                num_docs = len(corpus_tokens_by_docs)
                avg_doc_length = self.get_avg_doc_length()

                # Map each token to an array of indices of documents containing it
                token_to_doc_indices = defaultdict(set)
                for index, tokens in enumerate(corpus_tokens_by_docs):
                    for token in tokens:
                        token_to_doc_indices[token].add(index)

                unique_corpus_tokens = sorted(token_to_doc_indices.keys())
                self.vocabulary = unique_corpus_tokens

                num_tokens = len(unique_corpus_tokens)

                # Calculate the IDF vector using the BM25 Okapi formula
                df_vector = np.array(
                    [len(docs) for docs in token_to_doc_indices.values()],
                    dtype=np.float32,
                )
                idf_vector = np.log(
                    ((num_docs - df_vector + 0.5) / (df_vector + 0.5)) + 1
                )
                self.idf_vector = idf_vector

                # Calculate the TF matrix
                tf_matrix = np.zeros((num_docs, num_tokens), dtype=np.float32)

                # Map tokens to their "supposed" indices
                token_to_index = {
                    token: index for index, token in enumerate(unique_corpus_tokens)
                }
                self.token_to_index = token_to_index

                # Fill the TF matrix with token counts
                for index, tokens in enumerate(corpus_tokens_by_docs):
                    token_counter = Counter(tokens)
                    for token, count in token_counter.items():
                        tf_matrix[index, token_to_index[token]] = count

                # Calculate length for all documents
                doc_lengths_vector = np.array(
                    [len(doc) for doc in corpus_tokens_by_docs], dtype=np.float32
                )

                # Vectorized BM25 calculation
                theta_vector = doc_lengths_vector / avg_doc_length

                denominator = tf_matrix + (
                    self.k * (1 - self.b + self.b * theta_vector)
                ).reshape(-1, 1)

                bm25_matrix = idf_vector * ((tf_matrix * (self.k + 1)) / denominator)

                self.bm25_matrix = bm25_matrix.tolist()
            return self.bm25_matrix
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_reduced_matrix(self) -> list[list[float]] | None:
        try:
            if self.reduced_matrix is None:
                if self.bm25_matrix is None:
                    self.get_bm25_matrix()

                # Convert the BM25 matrix to a NumPy array
                bm25_matrix = np.array(self.bm25_matrix, dtype=np.float32)

                # Normalize and reduce the dimensionality of the matrix
                normalizer = Normalizer(norm="l2")
                svd = TruncatedSVD(n_components=NUM_COMPONENTS, random_state=42)

                pipeline = Pipeline([("normalize", normalizer), ("svd", svd)])

                # Fit the pipeline
                fitted_pipeline = pipeline.fit(bm25_matrix)
                reduced_matrix = fitted_pipeline.transform(bm25_matrix)

                # Store the fitted pipeline for later use
                self.pipeline = fitted_pipeline

                self.reduced_matrix = reduced_matrix.tolist()
            return self.reduced_matrix
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def cluster_documents(self) -> None:
        try:
            # Get the optimal number of clusters
            optimal_k = compute_optimal_k_clusters_elbow(self.get_reduced_matrix())

            """ optimal_k = compute_optimal_k_clusters_silhouette(
                self.get_reduced_matrix()
            ) """

            # Set up the k-means model
            k_means_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)

            # Run k-means clustering on the TF matrix
            k_means_model.fit(self.get_reduced_matrix())
            clusters = k_means_model.labels_

            # Assign cluster IDs
            documents: list[Document] = self.get_documents()

            for i, cluster_id in enumerate(clusters):
                documents[i].set_cluster_id(str(cluster_id))

            # Save k-means model
            self.k_means_model = k_means_model

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def vectorize_text(self, text: str) -> list[float] | None:
        try:
            avg_dl = self.get_avg_doc_length()

            # Clean and tokenize the text
            query_tokens = tokenize_and_clean_text(text)
            query_len = len(query_tokens)

            query_tf_counts = Counter(query_tokens)
            num_corpus_tokens = len(self.vocabulary)
            query_bm25_vector = np.zeros(num_corpus_tokens, dtype=np.float32)

            # Query length normalization factor part
            query_norm_factor_base = self.k * (1 - self.b + self.b * query_len / avg_dl)

            # Build BM25 vector based on corpus vocabulary
            for token, index in self.token_to_index.items():
                tf = query_tf_counts.get(token, 0)  # TF of token in query

                if tf > 0:
                    idf = self.idf_vector[index]  # Corpus IDF
                    numerator = tf * (self.k + 1)
                    denominator = tf + query_norm_factor_base
                    query_bm25_vector[index] = idf * (numerator / denominator)

            return query_bm25_vector.tolist()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def predict_cluster_for_query(self, query_text: str) -> str | None:
        try:
            # Vectorize the query and reshape it
            query_vector = np.array(self.vectorize_text(query_text)).reshape(1, -1)

            # Normalize and reduce the dimensionality of the query vector
            reduced_query_vector = self.pipeline.transform(query_vector)

            # Get the reduced matrix
            docs_matrix_reduced = np.array(self.get_reduced_matrix())

            # TODO: Use the one in QueriedBM25Corpus
            from sklearn.metrics.pairwise import cosine_similarity

            # Compute cosine similarities between the query vector and the document vectors
            similarities = cosine_similarity(reduced_query_vector, docs_matrix_reduced)[
                0
            ]

            # Get the indices of the nearest neighbors
            nearest_doc_indices = np.argsort(similarities)[-NUM_NEIGHBORS:]

            # Obtener los cluster IDs de los documentos vecinos
            neighbor_clusters = [
                self.documents[i].get_cluster_id() for i in nearest_doc_indices
            ]

            # Encontrar el cluster más frecuente entre los vecinos
            most_common_cluster = Counter(neighbor_clusters).most_common(1)

            if not most_common_cluster:
                print(
                    "Advertencia: No se encontraron clusters para los documentos vecinos"
                )
                if nearest_doc_indices.size > 0:
                    return self.documents[nearest_doc_indices[-1]].get_cluster_id()
                return None

            # Get the most common cluster ID
            return most_common_cluster[0][0]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


class QueriedDocument:
    def __init__(
        self,
        document: Document,
        score: float,
        token_score_pairs: list[tuple[str, float]],
    ):
        self.document: Document = document
        self.score: float = score
        self.token_score_pairs: list[tuple[str, float]] = token_score_pairs

    def get_document(self) -> Document:
        return self.document

    def get_score(self) -> float:
        return self.score

    def get_token_score_pairs(self) -> list[tuple[str, float]]:
        return self.token_score_pairs


class QueriedBM25Corpus:
    def __init__(self, corpus: Corpus, query_text: str):
        self.query_text: str = query_text
        self.query_tokens: list[str] | None = None
        self.corpus: Corpus = corpus
        self.queried_documents: list[QueriedDocument] = []
        self.cluster_id: str | None = None

        self._assign_bm25_scores()

    def get_corpus(self) -> Corpus:
        return self.corpus

    def _assign_bm25_scores(self) -> float:
        avg_doc_length = self.corpus.get_avg_doc_length()

        for document in self.corpus.documents:
            document_score = 0
            token_score_pairs = []
            for token in self.get_query_tokens():
                token_score = bm25_score(
                    token,
                    document.get_tokens(),
                    self.corpus.get_tokens_by_docs(),
                    avg_doc_length,
                    self.corpus.k,
                    self.corpus.b,
                )
                document_score += token_score
                token_score_pairs.append((token, token_score))
            self.queried_documents.append(
                QueriedDocument(document, document_score, token_score_pairs)
            )

    def get_query_tokens(self) -> list[str]:
        try:
            if self.query_tokens is None:
                self.query_tokens = tokenize_and_clean_text(self.query_text)
            return self.query_tokens
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_documents_by_score(self) -> list[QueriedDocument]:
        # Sort the documents by score in descending order
        return sorted(
            self.queried_documents,
            key=lambda queried_doc: queried_doc.get_score(),
            reverse=True,
        )

    def get_cosine_similarities(self, query_vector: list[float]) -> list[float]:
        try:
            # Convert lists to NumPy arrays for vectorized operations
            query_array = np.array(query_vector, dtype=np.float32)
            tf_matrix = np.array(self.corpus.get_bm25_matrix(), dtype=np.float32)

            # Compute operations
            cosine_similarities = np.dot(tf_matrix, query_array) / (
                np.linalg.norm(query_array) * np.linalg.norm(tf_matrix, axis=1)
            )

            # Convert the result back to a list and return
            return cosine_similarities.tolist()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_similar_documents(self, query_vector: list[float]) -> list[Document]:
        try:
            cosine_similarities = self.get_cosine_similarities(query_vector)
            sorted_indices = np.argsort(cosine_similarities)[::-1]
            similar_documents = [self.corpus.documents[i] for i in sorted_indices]
            return similar_documents
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_document_snippet_by_id(self, document_id: str) -> str | None:
        try:
            TOKEN_RANGE = 5

            # Get the index of the document in the corpus
            doc_index = self.corpus.get_document_index_by_id(document_id)

            # Check if the document exists in the corpus
            if doc_index is None:
                raise ValueError(
                    f"Document with ID '{document_id}' not found in the corpus."
                )

            # Get the document by index
            queried_document: QueriedDocument = self.queried_documents[doc_index]

            # Get top scoring token
            top_token, _ = max(
                queried_document.get_token_score_pairs(),
                key=lambda pair: pair[1],
                default=(None, 0),
            )

            doc_tokens = queried_document.get_document().get_tokens()

            # NOTE: Not a fan of this part, but its quicker than checking if the token is in the document

            # Get the index of the top scoring token in the document
            try:
                top_token_index = doc_tokens.index(top_token)
            except ValueError:
                top_token_index = -1

            # Get the snippet of text around the top scoring token with the token highlighted

            start = max(0, top_token_index - TOKEN_RANGE)
            end = min(len(doc_tokens), top_token_index + TOKEN_RANGE)

            return " ".join(
                [
                    f"**{token}**" if (start + i) == top_token_index else token
                    for i, token in enumerate(doc_tokens[start:end])
                ]
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def filter_documents_by_cluster_id(
        documents: list[QueriedDocument], cluster_id: str
    ) -> list[QueriedDocument]:
        return list(
            filter(
                lambda x: x.get_document().get_cluster_id() == cluster_id,
                documents,
            )
        )
