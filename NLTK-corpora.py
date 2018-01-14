## NLTK - read and classify custom corpora
## Template v.1

from nltk.corpus.reader import TaggedCorpusReader,WordListCorpusReader,ChunkedCorpusReader,PlaintextCorpusReader
from nltk.tokenize import SpaceTokenizer,sent_tokenize,word_tokenize,PunktSentenceTokenizer
from nltk.corpus import gutenberg


## Corpus example ############################
sample  = gutenberg.raw("bible-kjv.txt")
sent    = sent_tokenize(sample)

for x in range(5):
    print("Sentence - %s\n"% (sent[x]))
    print("Words - %s\n"% (nltk.word_tokenize(sent[x])))
    
## Reading corpora from a text files ##########
## No POS tags, chunks or categories ##########
reader      = PlaintextCorpusReader("/Users/atul/nltk_data/corpora/gutenberg",r'^.*\.txt')
files       = reader.fileids()
print("File IDs:",files); print("Number of files:", len(files))
print(reader.words(files[0]))
print(reader.sents(files[0]))


## Reading tagged corpora #####################
reader      = TaggedCorpusReader('/Users/atul/nltk_data',r'brown.pos',tagset='en-brown')
reader1     = TaggedCorpusReader('/Users/atul/nltk_data',r'brown.pos',word_tokenizer=SpaceTokenizer())

print(reader.words())
print(reader.sents())
print(reader.tagged_words())
print(reader.tagged_sents())
print(reader.tagged_words(tagset='universal')) ## Mapping tags to universal format, if tagset is not correct every TAG will have UNK


## Reading chunk corpora #######
reader      = ChunkedCorpusReader('/Users/atul/nltk_data',r'treebank.chunk',tagset='en-brown')
print(reader.chunked_words()) ## Word level structure
print(reader.chunked_sents()) ## Sentence level structure
print(reader.chunked_paras()) ## Paragraph level structure


## Reading classifed corpora ##################
## classification extracted using cat_pattern (from file name), or cat_dict or cat_file ######
from nltk.corpus.reader import CategorizedPlaintextCorpusReader

reader = CategorizedPlaintextCorpusReader('/Users/atul/nltk_data', r'movie_.*\.txt',cat_pattern=r'movie_(\w+)\.txt') ## Easiest is to read files for different category
reader.categories()
reader.fileids(categories=['neg'])
reader.fileids(categories=['pos'])
reader.fileids()


















