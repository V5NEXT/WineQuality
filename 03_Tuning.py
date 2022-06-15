from ast import Add
from gc import callbacks
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


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


def Batch_Normalization():
    # Define model
    # Add batch normalization layers
    model = keras.Sequential([
        layers.Dense(2 ** 10, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(2 ** 10, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
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
        verbose=1,  # suppress output since we'll plot the curves
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


# Batch_Normalization()

def FinalModelRegression():
    # Combine training set and validation set to train final model
    combined_dataset = data_prep.get_combined_dataset()
    ds_train, ds_valid, ds_test = data_prep.split_data_regression(
        combined_dataset)
    ds_train = ds_train.append(ds_valid)

    # Execute load_data() for prediction
    X_train, X_test, y_train, y_test = data_prep.load_data(ds_train, ds_test)

    # Training configuration
    BATCH_SIZE = 2 ** 8

    # Model Configuration
    UNITS = 2 ** 10
    ACTIVATION = 'relu'
    DROPOUT = 0.2

    # Build final model from scratch

    def dense_block(units, activation, dropout_rate, l1=None, l2=None):
        def make(inputs):
            x = layers.Dense(units)(inputs)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.BatchNormalization()(x)
            return x
        return make

    # Model
    inputs = keras.Input(shape=(13,))
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(inputs)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile in the optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Fit model (and save training history)
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=200,
        verbose=0,
    )

    from joblib import Parallel, delayed
    import joblib


# Save the model as a pickle in a file

    joblib.dump(model, '/regression_final_Model.pkl')

    # # Making predictions from test set
    # predictions = model.predict(X_test)

    # # Evaluate
    # model_score = mean_absolute_error(y_test, predictions)
    # print("Final model score (MAE):", model_score)


FinalModelRegression()
