import spacy
import os

try:
    spacy_de = spacy.load("de_core_news_sm")
except IOError:
    os.system("python -m spacy download de_core_news_sm")
    spacy_de = spacy.load("de_core_news_sm")

try:
    spacy_en = spacy.load("en_core_web_sm")
except IOError:
    os.system("python -m spacy download en_core_web_sm")
    spacy_en = spacy.load("en_core_web_sm")


spacy_en = spacy.load("en_core_web_sm")
doc = spacy_en("Apple is looking at buying U.K. startup for $1 billion            ")
for token in doc:
    print({token.text:token.idx})