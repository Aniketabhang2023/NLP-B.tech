import spacy
from collections import Counter
import math


nlp = spacy.load("en_core_web_sm")


paragraph = """The news mentioned here is fake. Audience do not encourage fake news. 
Fake news is false or misleading"""


sentences = [sent.text.strip() for sent in nlp(paragraph).sents]


docs_cleaned = []
for sent in sentences:
    doc = nlp(sent)
    words = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    docs_cleaned.append(words)


all_words = [word for doc_words in docs_cleaned for word in doc_words]
term_freq = Counter(all_words)
total_terms = sum(term_freq.values())
tf_normalized = {word: count / total_terms for word, count in term_freq.items()}


N = len(docs_cleaned)
idf = {}
for word in term_freq.keys():
    doc_count = sum(1 for doc_words in docs_cleaned if word in doc_words)
    idf[word] = math.log((N + 1) / (doc_count + 1)) + 1  


tf_idf = {word: tf_normalized[word] * idf[word] for word in term_freq.keys()}

print("Cleaned Documents:", docs_cleaned)
print("\nTerm Frequency (Counts):", dict(term_freq))
print("\nTerm Frequency (Normalized):", tf_normalized)
print("\nInverse Document Frequency (IDF):", idf)
print("\nTF-IDF:", tf_idf)
