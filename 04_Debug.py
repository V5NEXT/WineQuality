import pickle

# Cross Validation Classification Accuracy
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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
    matrix = confusion_matrix(Y_test, predicted)
    print(matrix)


def ClassificationReport():
    predicted = model.predict(X_test)
    report = classification_report(Y_test, predicted)
    print(report)


##################################################################################################
# ************************* Regression Model Evaluation ***************************************
##################################################################################################
