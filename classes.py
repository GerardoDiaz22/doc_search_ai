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
import matplotlib.pyplot as plt  # Import for the elbow method


class Document:
    def __init__(self, path_to_document: str):
        if not path_to_document.endswith(".pdf"):
            raise ValueError("The document must be a PDF file")

        self.path_to_document: str = path_to_document
        self.name: str = path_to_document.split("/")[-1].split(".")[0]
        self.id: str = str(uuid.uuid4())
        self.text: str | None = None
        self.tokens: list[str] | None = None
        self.cluster_id: Optional[int] = None  # ADDED THIS FIELD

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

    def set_cluster_id(self, cluster_id: int) -> None:  # ADDED THIS METHOD
        self.cluster_id = cluster_id

    def get_cluster_id(self) -> Optional[int]:  # ADDED THIS METHOD
        return self.cluster_id

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
    def __init__(self):
        self.documents: list[Document] = []
        self.tokens: list[list[str]] | None = None
        self.vectorizer: Optional[TfidfVectorizer] = None  # ADDED THIS FIELD
        self.kmeans_model: Optional[KMeans] = None  # ADDED THIS FIELD
        self.optimal_n_clusters: Optional[int] = (
            None  # Store the optimal number of clusters
        )

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

    def determine_optimal_number_of_clusters(
        self, tfidf_matrix, max_clusters=10, plot_elbow=False
    ):
        """
        Determines the optimal number of clusters using the elbow method.

        Args:
            tfidf_matrix: The TF-IDF matrix of the documents.
            max_clusters: The maximum number of clusters to try.
            plot_elbow: Whether to plot the elbow graph.

        Returns:
            The optimal number of clusters.
        """
        wcss = []  # Within-cluster sum of squares
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
            kmeans.fit(tfidf_matrix)
            wcss.append(kmeans.inertia_)

        if plot_elbow:
            plt.plot(range(1, max_clusters + 1), wcss, marker="o")
            plt.title("Elbow Method for Optimal k")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            plt.show()

        # Implement a more robust elbow detection (e.g., using the rate of change)
        # This is a simplified approach. More sophisticated methods exist.
        # Find the "elbow" where the rate of decrease in WCSS slows down significantly
        elbow_index = 0
        for i in range(1, len(wcss)):
            if i > 1 and (wcss[i - 1] - wcss[i]) < 0.1 * (
                wcss[0] - wcss[1]
            ):  # Adjust threshold as needed
                elbow_index = i
                break
        else:  # If the loop completes without breaking, it means there's no clear elbow
            elbow_index = len(wcss) - 1  # Take the last one

        optimal_clusters = elbow_index + 1
        print(
            f"Optimal number of clusters determined by elbow method: {optimal_clusters}"
        )
        return optimal_clusters

    def cluster_documents(
        self, n_clusters: Optional[int] = None
    ) -> None:  # ADDED THIS METHOD
        """Clusters the documents in the corpus using KMeans, determining the optimal number of clusters using the elbow method if n_clusters is not provided."""
        try:
            # 1. Prepare document texts
            document_tokens = [doc.get_tokens() for doc in self.documents]

            # 2.  Handle None tokens:  Important, skip documents with no tokens.
            valid_indices = [i for i, tokens in enumerate(document_tokens) if tokens]
            valid_tokens = [document_tokens[i] for i in valid_indices]
            valid_documents = [self.documents[i] for i in valid_indices]

            if not valid_tokens:
                print("Warning: No valid documents found for clustering.")
                return

            # 3. Vectorize the texts using TF-IDF.  Join the tokens into strings for TF-IDF.
            self.vectorizer = TfidfVectorizer()
            tfidf_matrix = self.vectorizer.fit_transform(
                [" ".join(tokens) for tokens in valid_tokens]
            )

            # 4. Determine optimal number of clusters if not provided.
            if n_clusters is None:
                self.optimal_n_clusters = self.determine_optimal_number_of_clusters(
                    tfidf_matrix
                )
            else:
                self.optimal_n_clusters = n_clusters  # Use the provided value

            # 5. Run KMeans clustering
            self.kmeans_model = KMeans(
                n_clusters=self.optimal_n_clusters, random_state=0, n_init=10
            )  # Setting n_init explicitly to suppress warning
            clusters = self.kmeans_model.fit_predict(tfidf_matrix)

            # 6. Assign cluster IDs to the documents (only to the valid ones)
            for i, cluster_id in enumerate(clusters):
                valid_documents[i].set_cluster_id(cluster_id)

            print(f"Clustered documents into {self.optimal_n_clusters} clusters.")
            self.display_cluster_composition()  # CALL THE PRINT FUNCTION
        except Exception as e:
            print(f"An error occurred during clustering: {e}")

    def get_documents_by_cluster(
        self, cluster_id: int
    ) -> List[Document]:  # ADDED THIS METHOD
        """Returns a list of documents belonging to a specific cluster."""
        return [doc for doc in self.documents if doc.get_cluster_id() == cluster_id]

    def predict_cluster_for_query(self, query_text: str) -> Optional[int]:
        """Predicts the most likely cluster for a given query."""
        try:
            if self.vectorizer is None or self.kmeans_model is None:
                raise ValueError("Corpus must be clustered first.")

            # Vectorize the query using the same vectorizer used for clustering
            query_vector = self.vectorizer.transform([query_text])

            # Predict the cluster
            cluster_id = self.kmeans_model.predict(query_vector)[0]
            return cluster_id
        except Exception as e:
            print(f"Error predicting cluster for query: {e}")
            return None

    def display_cluster_composition(self) -> None:
        """Prints the documents belonging to each cluster."""
        if self.kmeans_model is None:
            print("Corpus must be clustered first.")
            return

        num_clusters = self.kmeans_model.n_clusters
        print("\nCluster Composition:")
        for cluster_id in range(num_clusters):
            documents_in_cluster = self.get_documents_by_cluster(cluster_id)
            if documents_in_cluster:
                print(f"  Cluster {cluster_id}:")
                for doc in documents_in_cluster:
                    print(f"    - {doc.get_name()}")  # Print document name or ID
            else:
                print(f"  Cluster {cluster_id}: (Empty)")


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
    def __init__(
        self,
        corpus: Corpus,
        query_text: str,
        k: float,
        b: float,
        predicted_cluster_id: Optional[int] = None,
    ):  # MODIFIED THIS METHOD
        self.query_text: str = query_text
        self.query_tokens: list[str] | None = None
        self.k: float = k
        self.b: float = b
        self.corpus: Corpus = corpus
        self.queried_documents: list[QueriedDocument] = []
        self.tf_matrix = None
        self.predicted_cluster_id: Optional[int] = (
            predicted_cluster_id  # ADDED THIS FIELD
        )
        self._assign_bm25_scores()

    def get_corpus(self) -> Corpus:
        return self.corpus

    def _assign_bm25_scores(self) -> None:  # MODIFIED THIS METHOD
        avg_doc_length = self.corpus.get_avg_doc_length()

        # Use only documents from the predicted cluster, if a predicted_cluster_id is provided.
        if self.predicted_cluster_id is not None:
            documents_to_search = self.corpus.get_documents_by_cluster(
                self.predicted_cluster_id
            )
        else:
            documents_to_search = self.corpus.documents

        for document in documents_to_search:
            document_score = 0
            token_score_pairs = []
            for token in self.get_query_tokens():
                token_score = bm25_score(
                    token,
                    document.get_tokens(),
                    self.corpus.get_tokens_by_docs(),
                    avg_doc_length,
                    self.k,
                    self.b,
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

    def get_tf_matrix(self) -> list[list[float]] | None:
        try:
            if self.tf_matrix is None:
                corpus_tokens_by_docs: list[list[str]] = (
                    self.corpus.get_tokens_by_docs()
                )
                # Filter tokens to use only tokens from the cluster we're interested in
                if self.predicted_cluster_id is not None:
                    documents_in_cluster = self.corpus.get_documents_by_cluster(
                        self.predicted_cluster_id
                    )
                    corpus_tokens_by_docs = [
                        doc.get_tokens() for doc in documents_in_cluster
                    ]  # Overwrite with the cluster's docs.

                num_docs = len(corpus_tokens_by_docs)
                avg_doc_length = self.corpus.get_avg_doc_length()

                # Map each token to an array of indices of documents containing it
                token_to_doc_indices = defaultdict(set)
                for index, tokens in enumerate(corpus_tokens_by_docs):
                    for token in tokens:
                        token_to_doc_indices[token].add(index)

                unique_corpus_tokens = sorted(token_to_doc_indices.keys())
                num_tokens = len(unique_corpus_tokens)

                # Calculate the IDF vector using the BM25 Okapi formula
                df_vector = np.array(
                    [len(docs) for docs in token_to_doc_indices.values()],
                    dtype=np.float32,
                )
                idf_vector = np.log(
                    ((num_docs - df_vector + 0.5) / (df_vector + 0.5)) + 1
                )

                # Calculate the TF matrix
                tf_matrix = np.zeros((num_docs, num_tokens), dtype=np.float32)

                # Map tokens to their "supposed" indices
                token_to_index = {
                    token: index for index, token in enumerate(unique_corpus_tokens)
                }

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

                self.tf_matrix = bm25_matrix.tolist()
            return self.tf_matrix
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_cosine_similarities(self, query_vector: list[float]) -> list[float]:
        try:
            cosine_similarities = []
            tf_matrix = self.get_tf_matrix()
            if tf_matrix is None:
                return None  # Or handle the error as appropriate

            for doc_vector in tf_matrix:
                # Handle zero vector
                doc_vector = np.array(doc_vector)
                query_vector = np.array(query_vector)
                norm_doc_vector = np.linalg.norm(doc_vector)
                norm_query_vector = np.linalg.norm(query_vector)

                if norm_doc_vector == 0 or norm_query_vector == 0:
                    similarity = 0  # if one of the vector is a zero vector, cosine similarity will be zero.
                else:
                    similarity = np.dot(query_vector, doc_vector) / (
                        norm_query_vector * norm_doc_vector
                    )

                cosine_similarities.append(similarity)
            return cosine_similarities
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_similar_documents(self, query_vector: list[float]) -> list[Document]:
        try:
            cosine_similarities = self.get_cosine_similarities(query_vector)
            if cosine_similarities is None:
                return None

            sorted_indices = np.argsort(cosine_similarities)[::-1]

            # Ensure you're using the correct document set (clustered or full corpus)
            if self.predicted_cluster_id is not None:
                documents_in_cluster = self.corpus.get_documents_by_cluster(
                    self.predicted_cluster_id
                )
            else:
                documents_in_cluster = self.corpus.documents

            # Retrieve documents based on sorted indices, make sure not to go out of bounds
            similar_documents = [
                documents_in_cluster[i]
                for i in sorted_indices
                if i < len(documents_in_cluster)
            ]
            return similar_documents
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_document_snippet_by_id(self, document_id: str) -> Optional[str]:
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
            # This is where you have to choose between the original docments or the clustered ones.
            if self.predicted_cluster_id is not None:
                documents_in_cluster = self.corpus.get_documents_by_cluster(
                    self.predicted_cluster_id
                )
                # Find the index of the document in the cluster's document list
                try:
                    # doc_index_in_cluster = next(i for i, doc in enumerate(documents_in_cluster) if doc.get_id() == document_id)
                    queried_document: QueriedDocument = next(
                        qd
                        for qd in self.queried_documents
                        if qd.document.get_id() == document_id
                    )
                except StopIteration:
                    return "Document not found in cluster"
            else:
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
