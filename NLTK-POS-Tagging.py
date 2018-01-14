## NLTK - POS Tagging
## v1.2

import nltk
from nltk.corpus import state_union,stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk import RegexpParser 

## Data
train_text  = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

## Train tokenizer
custom_sent_tokenizer   = PunktSentenceTokenizer(train_text)
tokenized               = custom_sent_tokenizer.tokenize(sample_text)

## POS TAGGING AND CHUNKING
def process_tagging():
    
    try:
        for i in tokenized:
            words       = nltk.word_tokenize(i)
            tagged      = nltk.pos_tag(words)
            
            chunkGram   = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}
                                                                }<VB.?|IN|DT|TO>+{""" ## REGEX within {} will be caught (chunked) and REGEX between }{ will be removed (chincking) from chuncked
            chunkParser = nltk.RegexpParser(chunkGram)
            
            chunked     = chunkParser.parse(tagged)
            # chunked.draw()     

    except Exception as e:
        print(str(e))

## Run function
process_tagging()


### BIGRAM AND TRIGRAM TAGGERS #######################################
from nltk.corpus import treebank
from nltk.tag import UnigramTagger,BigramTagger,TrigramTagger

## Data
reader          = treebank
train_sents     = reader.tagged_sents()[:3000]
test_sents      = reader.tagged_sents()[3000:6000]

## Unigram tagger
tagger          = UnigramTagger(train_sents)
print("tagger accuracy:",tagger.evaluate(test_sents))

## Chaining taggers/ backoff tagging
tagger2         = BigramTagger(train_sents)
tagger3         = UnigramTagger(train_sents,backoff=tagger2)
print("tagger accuracy:",tagger3.evaluate(test_sents)) ## Chaining decreses accuracy - too much context is not good

## Chaining taggers- advanced mode
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff
    
backoff         = DefaultTagger('NN')
tagger          = backoff_tagger(train_sents, [UnigramTagger, BigramTagger,TrigramTagger], backoff=backoff)
tagger.evaluate(test_sents)







