## ML-DecTree
## Template v.1.2


#### Template 1 | Titanic data ####################################
###################################################################
import csv
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split

### Read Data
csvfile         = open('/Users/atul/Work/rodeo/titanic.csv', 'rb')
titanic_reader  = pd.read_csv(csvfile, delimiter=',',quotechar='"')
arow            = ["row.names","pclass","survived","name","age","embarked","home.dest","room","ticket","boat","sex"]
feature_names   = np.array(arow); print(feature_names)

# Load dataset, and target classes
titanic_X, titanic_y = [], []
for index, row in titanic_reader.iterrows():
    print(row)
    titanic_X.append(row)
    titanic_y.append(row[2]) # The target value is "survived"
    
titanic_X = np.array(titanic_X)
titanic_y = np.array(titanic_y)
print(titanic_X[0], titanic_y[0])

## Select Passanger class, age and sex as features from data and labels
titanic_X               = titanic_X[:, [1,4,10]]
feature_names           = feature_names[[1,4,10]];print(feature_names)


## Data imputation - Add missing ages
ages        = titanic_X[:, 1]
mean_age    = np.mean(titanic_X[ages != 'NA',1].astype(np.float))
titanic_X[titanic_X[:, 1] == 'NA', 1] = mean_age


# Encode sex i.e. convert to numerical values
from sklearn.preprocessing import LabelEncoder
enc             = LabelEncoder()
label_encoder   = enc.fit(titanic_X[:, 2])
print("Categorical classes:", label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)

t = label_encoder.transform(titanic_X[:, 2])
titanic_X[:, 2] = t


### One hot encoding - Conversion of feature values to new features with binary values
from sklearn.preprocessing import OneHotEncoder

enc             = LabelEncoder()
label_encoder   = enc.fit(titanic_X[:, 0])
print("Categorical classes:", label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3, 1)
print("Integer classes:", integer_classes)
enc             = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)

# First, convert classes to 0-(N-1) integers using label_encoder
num_of_rows     = titanic_X.shape[0]
t               = label_encoder.transform(titanic_X[:,0]).reshape(num_of_rows, 1)

# Second, create a sparse matrix with three columns, each one indicating if the instance belongs to the class
new_features    = one_hot_encoder.transform(t)

# Add the new features to titanix_X
titanic_X       = np.concatenate([titanic_X, new_features.toarray()], axis = 1)

# Eliminate converted columns
titanic_X       = np.delete(titanic_X, [0], 1)

# Update feature names
feature_names   = ['age', 'sex', 'first_class', 'second_class', 'third_class']

# Convert to numerical values
titanic_X       = titanic_X.astype(float)
titanic_y       = titanic_y.astype(float)

## Check
print (feature_names)
print (titanic_X[0], titanic_y[0])

## Holdout
X_train, X_test, y_train, y_test = train_test_split(titanic_X,titanic_y, test_size=0.25, random_state=33)