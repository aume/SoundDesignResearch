'''
learnTest.py /path/to/results_features.csv (File, Value, ...)
readm .csv file of study data ("Value" is the response Y variable)
1. pull out response and predictor variables
2. select best feature_selection
3. train and test SVR
4. save model
5. save selected feature names to file
'''

import pandas as pd
import sys

# read in data using pandas keeping column names
data = pd.read_csv(sys.argv[1])
data.pop('File') # remove file name
y = data.pop('Value').values # remove and capture response variable
feature_cols = data.columns.values
X = data.loc[:, feature_cols]

# evaluation of a model using 10 features chosen with correlation
# https://machinelearningmastery.com/feature-selection-for-regression-data/
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=f_regression, k=20)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = SVR(C=1, epsilon=0.1)
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
cols = fs.get_support(indices=True)
# Create new dataframe with only desired columns, or overwrite existing
print(feature_cols[cols])

import pickle
# save the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
filename = 'select_features.pkl'
pickle.dump(feature_cols[cols], open(filename, 'wb'))
# save selected feature names to file

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# loaded_model.predct()
