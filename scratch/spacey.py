import os

import spacy

# install spacey tokenizers for En and DE.
try:
    spacy_de = spacy.load("de_core_news_sm")
except IOError:
    os.system("uv run spacy download de_core_news_sm")
    spacy_de = spacy.load("de_core_news_sm")

try:
    spacy_en = spacy.load("en_core_web_sm")
except IOError:
    os.system("uv run spacy download en_core_web_sm")
    spacy_en = spacy.load("en_core_web_sm")


doc = spacy_en("Apple is looking at buying U.K. startup for $1 billion    ")
for token in doc:
    print({token.text:token.idx})


token.orth_


doc1 = spacy_en("yes   ")
doc2 = spacy_en("yes          ")

print([(token.text, token.lemma_) for token in doc1])
print([(token.text, token.lemma_) for token in doc2])

doc1[-1].orth
doc2[-1].orth