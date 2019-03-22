import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Churn_Modelling.csv')

x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lab1 = LabelEncoder()
x[:,1] = lab1.fit_transform(x[:,1])
lab2 = LabelEncoder()
x[:,2]= lab2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 0) 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu',input_dim = 11))
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dropout(p = 0.1))


classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train,y_train,batch_size = 10, epochs = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn .metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cmb = np.array(cma)
acc = (cm[0,0]+cm[1,1])/cm.sum()*100

print('you got ' ,acc,'% Accuracy')

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
