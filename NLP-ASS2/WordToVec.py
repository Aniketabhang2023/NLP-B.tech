import spacy


nlp = spacy.load("en_core_web_lg")

word1 = nlp("king")
word2 = nlp("queen")
word3 = nlp("man")
word4 = nlp("woman")


print("Similarity between king and queen:", word1.similarity(word2))
print("Similarity between man and woman:", word3.similarity(word4))


print("Vector for 'king':", word1.vector[:10]) 
