# How to run the app
- Install the requirements
```bash
pip install -r requirements.txt
```
- Run the app
```bash
streamlit run main.py
```

# Roadmap
- [x] Add a dataset for testing # Gerardo
- [x] Change all the by_name handlings to use by_index or by_id # Gerardo
- [x] Change the QueriedDocument to save the (token, score) pair # Gerardo
- [x] Fix get_snippet_from_doc_by_score to actually use the query # Gerardo
- [x] Add initialization step before showing the search bar # Gerardo
- [x] Add clustering with k-means # Jose
- [x] Improve UI
- [x] Add metrics to evaluate the performance of the model # Gerardo
- [ ] Write report document # Jose
- [ ] Add env variable handling for dev and prod # Gerardo
- [ ] Add clean text before lemmatization to the document class # Gerardo
- [ ] Check how are words with accents being handled by the tokenizer # Gerardo
- [ ] Add more aggresive tokenization # Gerardo
- [x] Change UI to select a document # Gerardo
- [x] Improve performance of get_similar_documents # Gerardo
- [x] Add clustering as filter for the search
- [x] Add requirements.txt # Gerardo
- [ ] Add steps to run the app in the README # Gerardo