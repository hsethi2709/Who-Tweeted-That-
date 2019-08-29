from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dropout
from keras.preprocessing import text, sequence
import numpy as np
import pandas as pd
import tensorflow as tf


# load dataset
file = "train_tweets.txt"
temp = []
with open(file, 'r') as data:
    for line in data:
        row = []
        line = line.replace('\t'," ")
        elem = line.strip().split(" ")
        row.append(elem[0])
        row.append(" ".join(elem[1:]))
        temp.append(row)

tw = pd.DataFrame(temp,columns = ["User","Tweet"])
X_train, X_test, y_train, y_test = train_test_split(tw.Tweet, tw.User, random_state=0, test_size=0.3)

print("Data Split")

max_words = 2000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(pd.Series(tw['Tweet'])) # only fit on train
x_train = tokenize.texts_to_matrix(X_train)
x_test = tokenize.texts_to_matrix(X_test)

print("Data tokenized")

encoder = LabelEncoder()
encoder.fit(pd.Series(tw['User']))
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

print("Data Encoded")

num_classes = np.max(y_train) + 1
#y_train = utils.to_categorical(y_train, num_classes)
#y_test = utils.to_categorical(y_test, num_classes)


print("Model being trained")


def baseline_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    #Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_crossentropy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=30, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
