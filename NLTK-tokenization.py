## Tokenization and Stopwords
## Template v.1

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import WordPunctTokenizer,PunktSentenceTokenizer
from nltk.corpus import webtext

#### TOKENIZE ######################################################################
## Using default (treebank) tokenizers
para = "Hello World. It's good to see you. Thanks for buying this book."
print(sent_tokenize(para))
print(word_tokenize(para)) ## punctation is treated as a seprate word

## Alternative word tokenizer
tokenizer = WordPunctTokenizer() ## Punctuation is a seprate word
print(tokenizer.tokenize(para))

## Make your sentence tokenizer - based on unsupervised learning
text            = webtext.raw('overheard.txt')  ## Read corpus example
sent_tokenizer1 = PunktSentenceTokenizer(text)  ## Train tokenizer
sent1           = sent_tokenizer1.tokenize(text)## Use new tokenizer
sent            = sent_tokenize(text)           ## Old tokenizer

#### check difference between tokenizers
print("Default tokenizer:\n",sent[678])     ## Fails to tokenize sentences properly
print("Learned tokenizer:\n",sent1[678])    ## Works well


#### STOPWORDS ######################################################################
from nltk.corpus import stopwords

## Find languages
stopwords.fileids() ## Which languages
stopwords.words('english')[1:10]

## Filtering stop words
stopset = set(stopwords.words('english'))
words   = ["Can't", 'is', 'a', 'contraction']
print([word for word in words if word not in stopset])