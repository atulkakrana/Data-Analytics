## NLTK - synsets
## Template

from nltk.corpus import wordnet


## Get synsets ####################
syns = wordnet.synsets("program")
print("All:",syns)
print("First one:",syns[0].name())
print("First one - lemmas",syns[0].lemmas()[0])
print(syns[0].examples())

## Get Antynoyms ##################
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

## Semnatic similarity
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1,w2); print(w1.wup_similarity(w2)) ## Wu and Palmer semantic similarity

print(synonyms[0].wup_similarity(synonyms[1])) ## Wu and Palmer semantic similarity

