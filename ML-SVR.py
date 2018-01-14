## Practical Machine Learning Tutorial - LinearRegression
## Link: https://pythonprogramming.net/machine-learning-tutorial-python-introduction/
## Youtube: https://www.youtube.com/watch?v=r4mwkS2T9aI&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=4

## LinearRegression #########
## SupportVectorMachine #####
import pandas as pd
import quandl,math,pickle
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key = 'PWmvYgQeEr2oAHM14jwR'

## Prepare data
df_main = quandl.get("WIKI/GOOGL")

df              = df_main[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']];df
df['HL_PCT']    = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0;df
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df              = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

## Prepare values that are to be precidted for this regression example
forecast_col    = 'Adj. Close'
df.fillna(value = -99999, inplace=True)
forecast_out    = int(math.ceil(0.01 * len(df)))            ## Adding 10-day (or something like that) as dummy to fit the regression
df['label']     = df[forecast_col].shift(-forecast_out)     ## Adding 10-day (or something like that) as dummy to fit the regression
print(df.head())

## Features and Labels
df.dropna(inplace=True)         ## Remove rows that do not have "y" data for learning
x = np.array(df.drop(['label'], 1)) ## Exclude "y" data from data frame
x = preprocessing.scale(x)      ## Used to model regression using dummy "y" data
y = np.array(df['label'])       ## Used to model regression using dummy "y" data
len(df)

## Split data
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

## Train regression
clf         = svm.SVR(kernel='linear') ## Support Vector Regression
clf         = clf.fit(x_train, y_train)
accuracy    = clf.score(x_test, y_test); print(accuracy) ## Accuracy from trained data

## Test
x_lately = x[-forecast_out:]            ## Slice out rows after the dummy "y" data is added 
forecast_set = clf.predict(x_lately)    ## Prediction
print(forecast_set, accuracy)

## Save your classifier for later use
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
## Read in future 
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

### END ################################################################


