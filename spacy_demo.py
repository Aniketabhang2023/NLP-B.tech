import spacy

nlp = spacy.load("en_core_web_sm")

introduction_doc = nlp("This tutorial is about Natural Language Processing in spaCy.")

tokens = [token.text for token in introduction_doc]

print(tokens)
