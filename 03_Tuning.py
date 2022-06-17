from ast import Add
from gc import callbacks
from pyexpat import model
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from gc import callbacks
import pandas as pd
import numpy as np
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)


data_model = __import__('02_Model')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)
##################################################################################################
#************************* Classification Model Tuning **********************************************#
##################################################################################################
# model callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=0)  # val_loss
callbacks = [early_stopping]

# Prediction of Wine Type


def Changing_Layers_Classification():
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

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.ylim([0, max([max(loss), max(val_loss)])])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    # plt.show()  Commented to check runtime

    val_loss_changing_layers = val_loss
    return val_loss_changing_layers, model


def ChangingNodes_Classification():
    input_shape = (11, )
    layer1 = 12
    layer2 = 9
    epochs = 10
    batch_size = 1
    optimizer = 'adam'

    history_frame, model = data_model.BaseClassificationModel(
        input_shape, layer1, layer2, epochs, batch_size, optimizer)

    return history_frame['val_loss'].min(), model


def ChangingBatchSize_Classification():
    input_shape = (11, )
    layer1 = 12
    layer2 = 9
    epochs = 10
    batch_size = 256
    optimizer = 'adam'

    history_frame, model = data_model.BaseClassificationModel(
        input_shape, layer1, layer2, epochs, batch_size, optimizer)

    return history_frame['val_loss'].min(), model


def ChangingOptimizer_Classification():
    input_shape = (11, )
    layer1 = 12
    layer2 = 9
    epochs = 10
    batch_size = 1
    optimizer = 'SGD'

    history_frame, model = data_model.BaseClassificationModel(
        input_shape, layer1, layer2, epochs, batch_size, optimizer)

    return history_frame['val_loss'].min(), model


def FinalModel_Classification():
    input_shape = (11, )
    layer1 = 12
    layer2 = 9
    epochs = 10
    batch_size = 1
    optimizer = 'adam'
    X_train, X_test, y_train, y_test, X_val, y_val = data_prep.split_dataset_classification()

    history_frame, model = data_model.BaseClassificationModel(
        input_shape, layer1, layer2, epochs, batch_size, optimizer)

    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("classification_model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("classification_model.h5")
    # print("Saved model to disk")

    # # # Predicting the Value
    # y_pred = model.predict(X_test)
    # print(y_pred)

    return history_frame['val_loss'].min(), model


##################################################################################################
#************************* Regression Model Tuning **********************************************#
##################################################################################################


EPOCHS = 300
BATCH_SIZE = 2 ** 8  # 256
X_train, X_valid, y_train, y_valid = data_prep.regressionPreprocess()
input_shape = [X_train.shape[1]]


def Two_Represntation_Layers():
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

    return history_frame['val_loss'].min(), model


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

    return history_frame['val_loss'].min(), model


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

    return history_frame['val_loss'].min(), model

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
    return history_frame['val_loss'].min(), model


def Model_with_Callbacks():
    # (Updated) Training Configuration
    EPOCHS = 3000

    # Defining callbacks
    early_stopping = EarlyStopping(
        patience=50,  # how many epochs to wait before stopping
        min_delta=0.001,  # minimium amount of change to count as an improvement
        restore_best_weights=True,
    )
    lr_schedule = callbacks.ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )
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
    return history_frame['val_loss'].min(), model


def AddingDropOut():
    # (Updated) Training Configuration
    EPOCHS = 3000

    # Defining callbacks
    early_stopping = EarlyStopping(
        patience=50,  # how many epochs to wait before stopping
        min_delta=0.001,  # minimium amount of change to count as an improvement
        restore_best_weights=True,
    )
    lr_schedule = callbacks.ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )
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
    return history_frame['val_loss'].min(), model


def Batch_Normalization():

    # (Updated) Training Configuration
    EPOCHS = 3000

    # Defining callbacks
    early_stopping = EarlyStopping(
        patience=50,  # how many epochs to wait before stopping
        min_delta=0.001,  # minimium amount of change to count as an improvement
        restore_best_weights=True,
    )
    lr_schedule = callbacks.ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )
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
    return history_frame['val_loss'].min(), model


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

    # serialize model to JSON
    model_json = model.to_json()
    with open("regression_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("regression_model.h5")
    print("Saved model to disk")
    # Convert the training history to a dataframe
    history_frame = pd.DataFrame(history.history)

    # Plot training history
    history_frame.loc[0:, ['loss', 'val_loss']].plot()
    history_frame.loc[0:, ['mae', 'val_mae']].plot()
    return history_frame['val_loss'].min(), model


def Classification_Tuning():
    print("CLassification Tuning")
    val_loss_layer, model_change_layer = Changing_Layers_Classification()
    val_loss_layer = val_loss_layer[0]
    val_loss_nodes, model_change_nodes = ChangingNodes_Classification()
    val_loss_batch, model_change_batch = ChangingBatchSize_Classification()
    val_loss_optimizers, model_change_optimizer = ChangingOptimizer_Classification()
    val_loss_final, final_model = FinalModel_Classification()

    if(val_loss_layer < val_loss_nodes) and (val_loss_layer < val_loss_batch) and (val_loss_layer < val_loss_optimizers) and (val_loss_layer < val_loss_final):
        Final_Model = model_change_layer
    elif(val_loss_nodes < val_loss_layer) and (val_loss_nodes < val_loss_batch) and (val_loss_nodes < val_loss_optimizers) and (val_loss_nodes < val_loss_final):
        Final_Model = model_change_nodes
    elif(val_loss_final < val_loss_layer) and (val_loss_final < val_loss_batch) and (val_loss_final < val_loss_optimizers) and (val_loss_final < val_loss_nodes):
        Final_Model = final_model
    elif(val_loss_optimizers < val_loss_layer) and (val_loss_optimizers < val_loss_batch) and (val_loss_optimizers < val_loss_nodes) and (val_loss_optimizers < val_loss_final):
        Final_Model = model_change_optimizer
    else:
        Final_Model = model_change_batch
    return Final_Model


def Regression_Tuning():
    Two_Represntation_Layers()
    One_Representaion_Layer()
    Three_Represntaion_Layer()
    OptimizedModel()
    Model_with_Callbacks()
    AddingDropOut()
    Batch_Normalization()
    FinalModelRegression()
