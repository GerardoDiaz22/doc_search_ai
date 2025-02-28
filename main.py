import pymupdf
import spacy


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
    print(tokens)

    # Join the cleaned tokens
    return " ".join(tokens)


# STEP 1.1: Read the PDF file
pdf_text = get_text_from_pdf("my_pdf.pdf")

# STEP 1.2: Remove special characters, convert to lowercase, remove stopwords, tokenize and lemmatize
cleaned_text = clean_text(pdf_text)

# Create a file to write the text to
out = open("output.txt", "wb")

# Write the text to the output file
out.write(cleaned_text.encode("utf8"))

# Close the output file
out.close()
