import nltk
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

worf_quote = "Sir, I protest. I am not a merry man!"
words_in_quote = word_tokenize(worf_quote)
print(words_in_quote)

stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words_in_quote if word.lower() not in stop_words and word.isalpha()]
print(filtered_words)
