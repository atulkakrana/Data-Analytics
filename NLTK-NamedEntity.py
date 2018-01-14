## NLTK - Named entity reconizition

import nltk
from nltk.corpus import state_union,stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk import RegexpParser 
from nltk.corpus import stopwords

## Read data
train_text  = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

## Tokenize data
custom_sent_tokenizer   = PunktSentenceTokenizer(train_text)
tokenized               = custom_sent_tokenizer.tokenize(sample_text)

## Name Entity chunk - Error rate is high; combine with "Collocations"
def process_tagging():
    english_stops = set(stopwords.words('english'))
    try:
        for i in tokenized:
            words       = nltk.word_tokenize(i)
            words_filter= [word for word in words if word not in english_stops]
            tagged      = nltk.pos_tag(words_filter)
            namedEnt    = nltk.ne_chunk(tagged, binary=True) ## Ex: (NE Martin/NNP Luther/NNP King/NNP) ## Improve accuracy by filtering on Nouns and than calling ne_chunk
            print(namedEnt)
            # print(tagged)

    except Exception as e:
        print(str(e))

## Run function
process_tagging()