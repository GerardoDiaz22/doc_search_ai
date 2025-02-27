import pymupdf
import spacy
import re


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


def clean_text(text):
    # Eliminación de caracteres especiales usando expresiones regulares.
    # Elimina todo lo que no sea alfanumérico o espacio.
    text = re.sub(r"[^\w\s]", "", text)

    # Conversión a minúsculas
    text = text.lower()

    # Tokenización y lematización con spaCy
    spacy_doc = nlp(text)

    # Eliminación de stopwords y lematización
    tokens = [token.lemma_ for token in spacy_doc if not token.is_stop]

    # Unir los tokens limpios
    return " ".join(tokens)


pdf_text = get_text_from_pdf("my_pdf.pdf")

cleaned_text = clean_text(pdf_text)

# Create a file to write the text to
out = open("output.txt", "wb")

# Write the text to the output file
out.write(cleaned_text.encode("utf8"))

# Close the output file
out.close()
