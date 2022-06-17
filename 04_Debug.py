# Cross Validation Classification Accuracy
import pandas
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean


data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

Tuning = __import__('03_Tuning')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

##################################################################################################
# ************************* Classification Model Evaluation ***************************************
##################################################################################################

    # # Making predictions from test set

X_train, X_test, y_train, y_test, X_val, y_val = data_prep.split_dataset_classification()


# load json and create model
json_file = open('classification_model.json', 'r')
loaded_classifcation_model_json = json_file.read()
json_file.close()
loaded_classifcation_model = model_from_json(loaded_classifcation_model_json)
# load weights into new model
loaded_classifcation_model.load_weights("classification_model.h5")
print("Loaded model from disk")


def Accuracy(model):
    predictions = model.predict(X_test)

    accuracy_score = metrics.accuracy_score(y_test, predictions.round(),
                                            normalize=True, sample_weight=None)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_test, predictions.round(),
                                                              sample_weight=None, adjusted=False)
    print("Accuracy :", accuracy_score*100)
    print("Balanced Accuracy : ", balanced_accuracy_score*100)


def ConfusionMatrix(model):
    predictions = model.predict(X_test)

    matrix = metrics.confusion_matrix(y_test, predictions.round(
    ), labels=None, sample_weight=None, normalize=None)
    print(matrix)


def ClassificationReport(model):
    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions.round())
    print(report)


def Precission_Accuracy(model):
    predictions = model.predict(X_test)

    avg_precision = metrics.average_precision_score(
        y_test, predictions.round(), average='macro', pos_label=1, sample_weight=None)
    print("Average Precision", avg_precision)
    # calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)

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
X_train_reg, X_test_reg, y_train_reg, y_test_reg = data_prep.load_data(
    ds_train, ds_test)


# load json and create model
json_file = open('regression_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("regression_model.h5")

print("Loaded model from disk")


def MeanAbsolute_Error(model):
    predictions_regression = model.predict(X_test_reg)

    # Evaluate
    model_score = mean_absolute_error(y_test_reg, predictions_regression)
    print("Final Regression model score (MAE):", model_score)


def MeanSquaredError(model):
    predictions_regression = model.predict(X_test_reg)

    # Evaluate
    model_score = mean_squared_error(y_test_reg, predictions_regression)
    print("Final Regression model score (MSE):", model_score)


def R2Matrics(model):
    predictions_regression = model.predict(X_test_reg)

    # Evaluate
    model_score = r2_score(y_test_reg, predictions_regression)
    print("Final Regression model score (R2):", model_score)


def MaxResidualError(model):
    predictions_regression = model.predict(X_test_reg)

    residual_error = metrics.max_error(y_test_reg, predictions_regression)
    print("Residual Error", residual_error)


def Evaluate_Regression_Model():
    regression_model = Tuning.Regression_Tuning()
    print("The Final Regression Model Results : ")
    R2Matrics(regression_model)
    MeanAbsolute_Error(regression_model)
    MeanSquaredError(regression_model)
    MaxResidualError(regression_model)


def Evaluate_Classification_Model():
    classification_model = Tuning.Classification_Tuning()
    print("The Final Classification Model Results : ")

    Accuracy(classification_model)
    ConfusionMatrix(classification_model)
    ClassificationReport(classification_model)
    Precission_Accuracy(classification_model)


# Evaluate_Regression_Model()
