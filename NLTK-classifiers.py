## NLTK - Text classifers
## scikit-learn classifers
## Template v.1

import nltk, random
from nltk.corpus import movie_reviews,stopwords
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold

############################ NLTK-classifiers ###############################
#############################################################################

## Prepare data and labels based on categories
reader = movie_reviews ## just as a habit to keep names same

reader.categories()
reader.words()
reader.sents()
# reader.tagged_words()

## Prepare data and labels for classification
adata  = [] ## Store data and labels
acount = 0  ## Count files

#### Prepare for test/train split - clean data for stopwords and punctuations
for acat in reader.categories():
    ## Get files for each category
    files = reader.fileids(acat)
    for afile in files:                                                 ## for every file in each category
        # print("Cat:%s | File:%s | Words:%s"% (category,afile,))
        adata.append((list(reader.words(afile)),acat))
        acount+=1
    print("Total files added:%s" % acount)

## Shuffle
random.shuffle(adata)

#### NLTK-classifier ########################################################
#### Select useful features i.e most often used words in the corpora

## Clean total words from corpora
stopset = set(stopwords.words('english'))
wordL   = [] ## Store all words from both categories
acount  = 0; bcount = 0
for w in reader.words():
    acount+=1
    if w not in stopset:
        wordL.append(w.lower())
        bcount+=1
print("Total words:%s | Filtered words:%s" % (acount,bcount))

## Top words - Lets use these as features - absoulute studpidity as I can simply use sci-kit
wordL           = nltk.FreqDist(wordL) ## Returns a dict with words and frequency
word_features1  = list(wordL.keys())[:3000]
wordL_famous    = wordL.most_common(3000)
word_features   = [aset[0] for aset in wordL_famous]

## Find P/A for this top words (features) in each review
## We will use this P/A for classification ('good' may appear is positive reviews)
def find_features(afilewords):
    words       = set(afilewords)
    features    = {}
    for w in word_features:
        features[w] = w in words ## word as key and boolean as value
    return features
    
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(review), category) for (review, category) in adata] ## Here P/A for top words is generated for each review

## Test/train split - holdout
training_set        = featuresets[:1900] ## Data from 1900 reviews used as training data
test_set            = featuresets[1900:] ## Data from 100 reviews used for testing

## classifer
classifier = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(classifier, test_set)*100)
print(classifier.show_most_informative_features(15))

## Saving classifier
fh_out           = open("naivebayes.pickle","wb")   ## Open a pickle file, wb = write in bytes
pickle.dump(classifier, fh_out)                     ## Save in pickle
fh_out.close()                                      ## close file

## Load classifier
fh_in           = open("naivebayes.pickle", "rb")   ## Open pickle file to read, rb = read in bytes
classifier      = pickle.load(fh_in)                ## load classifer
fh_in.close()                                       ## close pickle file


################################## sk-learn classifiers ##################################################

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy",nltk.classify.accuracy(MNB_clf, test_set))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)










#### Naive Bayes classification
## Prepare data - could have doen this above but analyses below forced to make a new loop
x = []
y=  []
for afile in adata:
    xdata,ylabel = afile
    x.append(adata)
    if y == "pos":
        y.append(0)
    else:
        y.append(1)
print("Data:%s | Label:%s" % (len(x),len(y)))

## Test/train - holdout
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2,random_state=11)
## Vectorize data and classify
clf = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB()),])

#### Cross-validation
def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    # print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

clfs = [clf]
for clf in clfs:
    evaluate_cross_validation(clf, x_train,  y_train, 5)


    
         



