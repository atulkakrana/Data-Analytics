## Machine learning - scikit learn 
## Channel: https://www.youtube.com/watch?v=URTZ2jKCgBc&list=PLQVvvaa0QuDd0flgGphKCej-9jp-QdzZ3&index=1
## Test Map: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

## BASIC DIGITS EXAMPLE ###########################
###################################################

## Link: https://www.youtube.com/watch?v=KTeVOb8gaD4&index=2&list=PLQVvvaa0QuDd0flgGphKCej-9jp-QdzZ3
from sklearn import datasets,svm,tree
import matplotlib.pyplot as plt

## data ############################################
digits = datasets.load_digits()
print(digits.keys())
print(digits.data[:-1])     ## 1797 digits with different pixel combinations
print(digits.target)        ## 1797 targets b/w 0 and 9
print(digits.target_names)
print(digits.images[0])

## Make a classifier ###############################
clf         = svm.SVC(gamma=0.001,C=100) ## gamma is Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’; lesser improves accuracy
features    = digits.data[:-1]
labels      = digits.target[:-1]
clf         = clf.fit(features,labels)

## Test the classifer ##############################

## Plot test image
test_pos    = -1 ## index that we will use for data
plt.gray(); plt.matshow(digits.images[test_pos]); plt.show() 
## test prediction results
print ("Prediction:",clf.predict(digits.data[[test_pos]]))

