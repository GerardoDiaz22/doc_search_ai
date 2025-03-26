import pymupdf
import os
import numpy as np
import uuid
from utils import tokenize_and_clean_text, bm25_score


class Document:
    def __init__(self, path_to_document):
        if not path_to_document.endswith(".pdf"):
            raise ValueError("The document must be a PDF file")

        self.path_to_document: str = path_to_document
        self.name: str = path_to_document.split("/")[-1].split(".")[0]
        self.id: str = str(uuid.uuid4())
        self.text: str | None = None
        self.tokens: list[str] | None = None

    def get_path(self) -> str:
        return self.path_to_document

    def get_text(self) -> str | None:
        try:
            if self.text is None:
                self.text = self._read_text_from_pdf(self.path_to_document)
            return self.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_tokens(self) -> list[str] | None:
        try:
            if self.tokens is None:
                self.tokens = tokenize_and_clean_text(self.get_text())
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

    @staticmethod
    def _read_text_from_pdf(path_to_pdf: str) -> str | None:
        try:
            # Open the PDF file
            pdf_doc = pymupdf.open(path_to_pdf)

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
            if not os.path.exists(path_to_dir):
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
    def __init__(self, corpus: Corpus, query_text: str, k: float, b: float):
        self.query_text: str = query_text
        self.query_tokens: list[str] | None = None
        self.k: float = k
        self.b: float = b
        self.corpus: Corpus = corpus
        self.queried_documents: list[QueriedDocument] = []
        self.tf_matrix = None
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
                avg_doc_length = self.corpus.get_avg_doc_length()
                unique_corpus_tokens = set(
                    token for doc in corpus_tokens_by_docs for token in doc
                )

                tf_matrix = np.zeros(
                    (len(self.corpus.documents), len(unique_corpus_tokens)), dtype=int
                )

                for i, document in enumerate(self.corpus.documents):
                    for j, token in enumerate(unique_corpus_tokens):
                        tf_matrix[i][j] = bm25_score(
                            token,
                            document.get_tokens(),
                            corpus_tokens_by_docs,
                            avg_doc_length,
                            self.k,
                            self.b,
                        )
                self.tf_matrix = tf_matrix.tolist()
            return self.tf_matrix
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_cosine_similarities(self, query_vector: list[float]) -> list[float]:
        try:
            cosine_similarities = []
            for doc_vector in self.get_tf_matrix():
                similarity = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                )
                cosine_similarities.append(similarity)
            return cosine_similarities
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
