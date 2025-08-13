import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. 
Fake news is false or misleading"""

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

corpus = []
for sent in sentences:
    sent = re.sub('[^a-zA-Z]', ' ', sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(sent))

print("Cleaned Corpus:", corpus)

all_words = []
for sentence in corpus:
    all_words.extend(sentence.split())

term_freq = Counter(all_words)
total_terms = sum(term_freq.values())
tf_normalized = {word: count / total_terms for word, count in term_freq.items()}

print("\nTerm Frequency (Counts):", dict(term_freq))
print("\nTerm Frequency (Normalized):", tf_normalized)
