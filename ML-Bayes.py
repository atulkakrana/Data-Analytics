## Text classification template
## Template v.1.1

## Data
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics

#### Data
news = fetch_20newsgroups(subset='all')

#### Data exploration
print("News dataset:",news.keys())
print("Data:",news.data[0]); print("data dim:",len(news.data));
print("Label:",news.target[0],"| Label name:",news.target_names[0])

#### Split data
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=0) ## Holdout approach

clf_1 = Pipeline([('vect', CountVectorizer()),('clf', MultinomialNB())])
clf_2 = Pipeline([('vect', HashingVectorizer(non_negative=True)),('clf', MultinomialNB()),])
clf_3 = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB()),])


#### Cross-validation - sci-kit PAKT publisher guide for adding regex and stop words to vectorizer
def evaluate_cross_validation(clf, X, y, K):
    cv      = KFold(len(y), K, shuffle=True, random_state=0)     # create a k-fold croos validation iterator of k=5 folds
    scores  = cross_val_score(clf, X, y, cv=cv)              # by default the score used is the one returned by score method of the estimator (accuracy)
    print(scores)

clfs = [clf_1, clf_2, clf_3]
for clf in clfs:
    evaluate_cross_validation(clf, x_train,  y_train, 5)
    

#### Test classifer
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    
train_and_evaluate(clf_3, x_train, x_test, y_train, y_test)






