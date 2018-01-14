## scikit-learn SVM CLassifers 
## Template v.1 - Cancer, Iris, Face recongnization

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

#### Template - 1 | Cancer data ##########################################
##########################################################################

df = pd.read_csv('breast-cancer-wisconsin.data.txt');print(df.head())
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1));x[1]
y = np.array(df['class'])

## Holdout
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

## Generate classifier
clf         = svm.SVC()
clf         = clf.fit(x_train, y_train)
confidence  = clf.score(x_test, y_test)
print(confidence)

## Test classifer
example_measures    = np.array([[4,2,1,1,1,2,3,2,1]]) ## A test entry matching the imported data
example_measures    = example_measures.reshape(len(example_measures), -1);example_measures
prediction          = clf.predict(example_measures)
print(prediction)

## Template - 2 | IRIS data ###############################################
###########################################################################

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics

## Get Data
iris = datasets.load_iris()

print("Keys:",iris.keys())
print("Target:\n",iris.data[:5])
print("Target:\n",iris.target)
print("Target Names:\n",iris.target_names)

x_iris, y_iris  = iris.data, iris.target     ## Get data and labels
x, y = x_iris[:, :2], y_iris; print (x, y)  ## Using just first two data coulmns for training 
print(x[0], y[0])

## holdout and pre-process data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25, random_state=33)
print("Train data:\n",x_train[:5]); print("Train Lables:\n",y_train[:5])

## Standardize the features - Scaling and transformation
x_scaler        = preprocessing.StandardScaler().fit(x_train)   ## For each feature, calculate the average, subtract the mean
                                                                ## value from the feature value, and divide the result by their standard deviation.
print(x_scaler)
x_train         = x_scaler.transform(x_train);print(x_train[:5])
x_test          = x_scaler.transform(x_test);print(y_train[:5])

## Build, train and test classsifer
clf         = svm.SVC()
clf         = clf.fit(x_train, y_train)
confidence  = clf.score(x_test, y_test); print("Accuracy:",confidence)

## Prediction metrics
print(metrics.accuracy_score(y_test, y_test_pred))          ## Compare predicted labels with true labels
print(metrics.classification_report(y_test, y_test_pred,target_names=iris.target_names))
print(metrics.confusion_matrix(y_test, y_test_pred))

## Exploration 
import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = x_train[:, 1][y_train == i]
    ys = x_train[:, 2][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
    plt.legend(iris.target_names)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    
    
## CrossValidation
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline

# create a composite estimator made by a pipeline of the standarization and the linear model
clf     = Pipeline([('scaler', preprocessing.StandardScaler()),('linear_model', svm.SVC())])

# create a k-fold cross validation iterator of k=5 folds
cv      = KFold(x.shape[0], 5, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores  = cross_val_score(clf, x, y, cv=cv)
print(scores)


#### Template-3 | FACE CLASSIFICATION #########################################
###############################################################################

import sklearn as sk
from scipy.stats import sem
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVCfrom sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold

#### Get Data
faces = fetch_olivetti_faces() ## 400 images of 40 person
print(faces.DESCR)

#### Data exploration
print("faces dataset:",faces.keys())
print("images data dim:",faces.images.shape)
print("numerical data dim:",faces.data.shape)
print("Labels:",faces.target.shape)

#### Data pre-processing - Data is b/w 0 to 1 so no scaling/transformation required
print(np.max(faces.data))
print(np.min(faces.data))
print(np.mean(faces.data))

#### Print images
def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[],yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

## use above function
print_faces(faces.images, faces.target, 40)

#### Split data (or holdout)
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0) ## Holdout

#### Prepare classifer
clf = svm.SVC(kernel='linear')

#### Cross-validation
def cross_validation(clf, x, y, k):
    cv      = KFold(len(y), k, shuffle=True, random_state=0) ##  create a k-fold cross validation iterator
    scores  = cross_val_score(clf, x, y, cv=cv) ## by default the score used is the one returned by score method of the estimator (accuracy)
    print(scores)

cross_validation(clf, x_train, y_train, 5)

#### Train and Evaluate
from sklearn import metrics
def train_and_evaluate(clf, x_train, x_test, y_train, y_test):
    
    ## Train
    clf.fit(x_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(x_train, y_train))
    
    ## Evaluate
    y_pred = clf.predict(x_test)
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    
train_and_evaluate(clf, x_train, x_test, y_train, y_test)
