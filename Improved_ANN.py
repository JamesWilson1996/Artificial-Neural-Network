# Artificial Neural Network

# Part 1 - Data preprocessing

# Import Libraries
import numpy as np
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # :-1 doesn't include the last index which in this case is the DV
y = dataset.iloc[:, 13].values # -1 is the last index in a dataframe

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Make the Artificial Neural Network

# Import Keras and packages
import keras
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Add a time callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Initialise ANN
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(rate=0.1))

# Adding another hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

# Add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
"""
Dealing with DV with more than 1 category then change units to match categories
and change activation to softmax
"""

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""
More than 2 categories:
loss = categorical_crossentropy
"""

# Fitting the ANN to the training set
time_callback = TimeHistory()
classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=100, callbacks=[time_callback])
times = time_callback.times

# Part 3 - Making predictions

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Predicting a single new observation
"""
Predict if new customer will leave or stay
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Active Member: Yes
Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction>0.5)

# Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
Test_Accuracy = (cm[0, 0] + cm[1, 1]) / 2000

# Part 4 - Evaluating, Improving and Tuning

# Evaluating
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    """
    Dealing with DV with more than 1 category then change units to match categories
    and change activation to softmax
    """
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    """
    More than 2 categories:
    loss = categorical_crossentropy
    """
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
mean = accuracies.mean()
var = accuracies.std()
    
# Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    """
    Dealing with DV with more than 1 category then change units to match categories
    and change activation to softmax
    """
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    """
    More than 2 categories:
    loss = categorical_crossentropy
    """
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25, 32],
              'epochs':[100, 500],
              'optimizer':['adam', 'rmsprop']}
gridsearch = GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring= 'accuracy',
                          cv=10)
gridsearch = gridsearch.fit(X_train, y_train)
best_param = gridsearch.best_params_
best_acc = gridsearch.best_score_