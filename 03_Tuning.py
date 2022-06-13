from ast import Add
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

##################################################################################################
#************************* Classification Model Tuning **********************************************#
##################################################################################################


##################################################################################################
#************************* Regression Model Tuning **********************************************#
##################################################################################################

data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

EPOCHS = 300
BATCH_SIZE = 2 ** 8  # 256
X_train, X_valid, y_train, y_valid = data_prep.regressionPreprocess()
input_shape = [X_train.shape[1]]


def Sequential_Two():
    # Define model with two representation layers
    model = keras.Sequential([
        layers.Dense(2 ** 4, activation='relu', input_shape=input_shape),
        layers.Dense(2 ** 4, activation='relu'),
        layers.Dense(1),
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
    history_frame.loc[0:, ['mae', 'val_mae']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))


def One_Representaion_Layer():
    # Define model with one representation layer
    model = keras.Sequential([
        layers.Dense(2 ** 4, activation='relu', input_shape=input_shape),
        layers.Dense(1),
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
    history_frame.loc[0:, ['mae', 'val_mae']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))


def Three_Represntaion_Layer():
    # Define model with three representation layers
    model = keras.Sequential([
        layers.Dense(2 ** 4, activation='relu', input_shape=input_shape),
        layers.Dense(2 ** 4, activation='relu'),
        layers.Dense(2 ** 4, activation='relu'),
        layers.Dense(1),
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
    history_frame.loc[0:, ['mae', 'val_mae']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))


# Optimizing Model Architecture
def OptimizedModel():
    # Define new model (use previous baseline model as comparison)
    model = keras.Sequential([
        layers.Dense(2 ** 10, activation='relu', input_shape=input_shape),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(1),
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
    history_frame.loc[0:, ['mae', 'val_mae']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))


# (Updated) Training Configuration
EPOCHS = 3000

# Defining callbacks
early_stopping = callbacks.EarlyStopping(
    patience=50,  # how many epochs to wait before stopping
    min_delta=0.001,  # minimium amount of change to count as an improvement
    restore_best_weights=True,
)
lr_schedule = callbacks.ReduceLROnPlateau(
    patience=0,
    factor=0.2,
    min_lr=0.001,
)


def Model_with_Callbacks():
    # Define model
    # Add callbacks
    model = keras.Sequential([
        layers.Dense(2 ** 10, activation='relu', input_shape=input_shape),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dense(1),
    ])

    # Compile in the optimizer and loss functions
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
    history_frame.loc[0:, ['mae', 'val_mae']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))


def AddingDropOut():
    # Define model
    # Add dropout layers
    model = keras.Sequential([
        layers.Dense(2 ** 10, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])

    # Compile in the optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Fit model (and save training history)
    # (Add callbacks)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping, lr_schedule],
        verbose=0,  # suppress output since we'll plot the curves
    )

    # Convert the training history to a dataframe
    history_frame = pd.DataFrame(history.history)

    # Plot training history
    history_frame.loc[0:, ['loss', 'val_loss']].plot()
    history_frame.loc[0:, ['mae', 'val_mae']].plot()

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))
