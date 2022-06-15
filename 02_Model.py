from gc import callbacks
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score
import joblib
import pickle
import os

# Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import SGDClassifier

# from sklearn.preprocessing import MinMaxScaler

data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

X_train, X_val, X_test, y_train, y_val, y_test = data_prep.split_dataset()


def plotValLossAndLoss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.ylim([0, max([max(loss), max(val_loss)])])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
##################################################################################################
#************************* Classification Base Model  **********************************************#
##################################################################################################


# model callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=0)  # val_loss


callbacks = [early_stopping]

# Prediction of Wine Type


def BaseClassificationModel():
    # Import `Sequential` from `keras.models`
    from keras.models import Sequential

    # Import `Dense` from `keras.layers`
    from keras.layers import Dense
    X_train, X_test, y_train, y_test, X_val, y_val = data_prep.split_dataset_classification()

    # Initialize the constructor
    model = Sequential()

    # Add an input layer
    model.add(Dense(12, activation='relu', input_shape=(11, )))

    # Add one hidden layer
    model.add(Dense(9, activation='relu'))

    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))

    # Model output shape
    model.output_shape

    # Model summary
    model.summary()

    # Model config
    model.get_config()

    # List all weight tensors
    model.get_weights()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # Training Model
    history = model.fit(X_train, y_train, epochs=10,
                        batch_size=1, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks, shuffle=True)

    # # Predicting the Value
    # y_pred = model.predict(X_test)
    # print(y_pred)


BaseClassificationModel()

##################################################################################################
#************************* Regression Base Model **********************************************#
##################################################################################################
EPOCHS = 300
BATCH_SIZE = 2 ** 8  # 256
X_train, X_valid, y_train, y_valid = data_prep.regressionPreprocess()
input_shape = [X_train.shape[1]]


def BaseModel():

    # Training Configuration

    # Define linear model
    model = keras.Sequential([
        layers.Dense(1, input_shape=input_shape),
    ])

    # Compile in the optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Fit model (and save training history)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,  # suppress output since we'll plot the curves
    )

    # Convert the training history to a dataframe
    history_frame = pd.DataFrame(history.history)

    # Plot training history
    history_frame.loc[0:, ['loss', 'val_loss']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))
