import pymupdf
import spacy
from spacy.lang.es.examples import sentences 

# Load spaCy spanish model
nlp = spacy.load("es_core_news_sm")

doc = pymupdf.open("my_pdf.pdf") # open a document
out = open("output.txt", "w", encoding="utf-8") # create a text output
for page in doc: # iterate the document pages
    text = page.get_text() # get plain text (is in UTF-8)
    # Process the text using spaCy
    text = text.lower()
    doc_final = nlp(text)
    
    # Remove stopwords
    filtered_words = [token.text for token in doc_final if not token.is_stop]
    
    # Join the filtered words to form a clean text
    clean_text = ' '.join(filtered_words)
    
    print("Original Text:", text)
    print("Original Text: --------------------------------------------------------------")

    print("Text after Stopword Removal:", clean_text)

    out.write(clean_text) # write text of page
    out.write(bytes((12,))) # write page delimiter (form feed 0x0C)

out.close()