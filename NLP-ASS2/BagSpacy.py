import spacy
from collections import Counter


nlp = spacy.load("en_core_web_sm")

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. 
Fake news is false or misleading"""


doc = nlp(paragraph)


words = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]


term_freq = Counter(words)


total_terms = sum(term_freq.values())
tf_normalized = {word: count / total_terms for word, count in term_freq.items()}

print("Cleaned Words:", words)
print("\nTerm Frequency (Counts):", dict(term_freq))
print("\nTerm Frequency (Normalized):", tf_normalized)
