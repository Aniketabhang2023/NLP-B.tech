from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

stemmer = PorterStemmer()

string_for_stemming = """
Artificial intelligence is transforming industries.
Machines learning from data are making intelligent decisions."""

words = word_tokenize(string_for_stemming)

stemmed_words = [stemmer.stem(word) for word in words]

print("Original words:", words)
print("Stemmed words:", stemmed_words)
