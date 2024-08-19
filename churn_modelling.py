import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

dataset.head()

"""## divide dataset into independent and dependent features"""

# credit score to estimated salary is independent and exited is dependent
X = dataset.iloc[:,3:-1]
X

# dependent valriable
y=dataset.iloc[:,-1]
y

## Feature engineering
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

X.head()

X.drop(['Geography','Gender'],axis=1,inplace=True) # drop these column not rows
X.head()

# concat one hot encoded variables in X
X=pd.concat([X,geography,gender],axis=1)
X.head()

"""# splitting the dataset into training set and test set"""

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# Here, 0.2 means 20% of the data will be used for testing, and the remaining 80% will be used for training.

"""# Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train.shape

"""# create ANN"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,PReLU,ELU,ReLU,Dropout

## initialize ANN
classifier = Sequential()

## adding input layer
classifier.add(Dense(units=11,activation='relu'))

## adding first hidden layer and dropout to prevent overfitting
classifier.add(Dense(units=7,activation='relu'))
classifier.add(Dropout(0.2))
## adding second hidden layer
classifier.add(Dense(units=6,activation='relu'))
classifier.add(Dropout(0.3))

## adding the output layer
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) # adam default learning rate=0.01

"""## Early Stopping"""

early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=1000,callbacks=early_stopping)

## summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

"""# Predections and evaluating the model"""

# predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

## make the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
cm
score=accuracy_score(y_test,y_pred)
score