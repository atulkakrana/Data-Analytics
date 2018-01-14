from sklearn import tree,datasets,svm
features    = [[140,1],[130,1],[150,0],[170,0]] ## Weight and 1: Smooth and 0:Bumpy
labels      = [1,1,0,0]                         ## 1: Apple and 0: Orange
clf         = tree.DecisionTreeClassifier()     ## Empty classifier
clf         = clf.fit(features,labels)          ## Training
print (clf.predict([[160,0]]))                  ## weight =160 and bumpy - is it apple(1) or orange(0)


## P2: https://www.youtube.com/watch?v=KTeVOb8gaD4&index=2&list=PLQVvvaa0QuDd0flgGphKCej-9jp-QdzZ3
import matplotlib.pyplot as plt
from sklearn import tree,datasets,svm

## data ##################
digits = datasets.load_digits()
print(digits.keys())
print(digits.data[:-1])
print(digits.target[:-1])
print(digits.images[0])

## Make a classifier ####
clf         = svm.SVC(gamma=0.001,C=100) ## gamma is Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’; lesser improves accuracy
features    = digits.data[:-1]
labels      = digits.target[:-1]
clf         = clf.fit(features,labels)

## Test the classifer ###
## Plot test image
test_pos    = -1 ## index that we will use for data
plt.gray(); plt.matshow(digits.images[test_pos]); plt.show() 
## test prediction results
print ("Prediction:",clf.predict(digits.data[[test_pos]]))



