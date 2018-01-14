## NLTK - Reducing data/text size
## Template v.1

#### STEMMING AND LEMMITIZING #######################################################
from nltk.stem import PorterStemmer

atext        = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words        = nltk.word_tokenize(atext);print("All words:",words)
stopset      = set(stopwords.words('english'))
words_filter = [word for word in words if word not in stopset];print("Filtered words:",words_filter)
print("Filtered:",words_filter)

## Reducing words based on stem
stemmer     = PorterStemmer()
stemS       = set()
for w in words_filter:
    ws = stemmer.stem(w)
    stemS.add(ws)
print("Stemmed:",stemS)

## Reducing words based on lemmas - better than stemming as it always gives back a valid word
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmaS     = set()
for w in words_filter:
    ws = lemmatizer.lemmatize(w)
    lemmaS.add(ws)
print("Lemmatized:",lemmaS)

#### SYNSETS ########################################################################
## See NLTK-synsets.py for more #####################################################
from nltk.corpus import wordnet
syn = wordnet.synsets('book')[0]                                                     ## Synset is a group of words with same meaning
print(syn.hypernyms(),syn.hypernyms()[0].hyponyms(),syn.definition(),syn.examples()) ## Synsets are organized in a tree of close meaning words (hypernyms) 
                                                                                     ## and broad similarity words (hyponyms)
                                                                                     
## Part of speech (pos) | Noun (n), Adjective (a), Adverb (r) and verb (v)
print(wordnet.synsets('book', pos='a'))
## Lemmas are canonical/morphological form of same word
lemmas = syn.lemmas()
print([lemma.name() for lemma in lemmas])


#### COLLOCATIONS - words that appear together. Ex: United airlines #################
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

## Get all words and filter stop words
words   = [w.lower() for w in webtext.words('grail.txt')]
bcf     = BigramCollocationFinder.from_words(words)         ## Find collocations
print(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4))   ## Not good results


stopset = set(stopwords.words('english'));print(stopset[1:10])  ## Get stopsets
filter_stops = lambda w: len(w) < 3 or w in stopset
bcf.apply_word_filter(filter_stops)                             ## remove collations with stop words
print(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4))
####################################################################################



