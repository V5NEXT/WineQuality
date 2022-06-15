
# Cross Validation Classification Accuracy
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

##################################################################################################
# ************************* Classification Model Evaluation ***************************************
##################################################################################################


def ClassificationAccuracy():
    scoring = 'accuracy'
    results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


def LogLoss():
    scoring = 'neg_log_loss'
    results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))


def ROCAUC():
    scoring = 'roc_auc'
    results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring=scoring)
    print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


def ConfusionMatrix():
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    print(matrix)


def ClassificationReport():
    predicted = model.predict(X_test)
    report = classification_report(y_test, predicted)
    print(report)


def Precission_Accuracy():

    # calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    # create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    # add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.show()
    ##################################################################################################
    # ************************* Regression Model Evaluation ***************************************
    ##################################################################################################


    # # Making predictions from test set
combined_dataset = data_prep.get_combined_dataset()
ds_train, ds_valid, ds_test = data_prep.split_data_regression(
    combined_dataset)
ds_train = ds_train.append(ds_valid)
X_train, X_test, y_train, y_test = data_prep.load_data(ds_train, ds_test)


# load json and create model
json_file = open('regression_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("regression_model.h5")
predictions = loaded_model.predict(X_test)

print("Loaded model from disk")


def MeanAbsolute_Error():
    # Evaluate
    model_score = mean_absolute_error(y_test, predictions)
    print("Final Regression model score (MAE):", model_score)


def MeanSquaredError():
    # Evaluate
    model_score = mean_squared_error(y_test, predictions)
    print("Final Regression model score (MSE):", model_score)


def R2Matrics():
    # Evaluate
    model_score = r2_score(y_test, predictions)
    print("Final Regression model score (R2):", model_score)


R2Matrics()
