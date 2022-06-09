import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import MinMaxScaler


data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

X_train, X_val, X_test, y_train, y_val, y_test = data_prep.split_dataset()


scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


pipeline_SVM = Pipeline([("scaler", MinMaxScaler()),
                         ("pipeline_SVM", SVC())])

pipeline_KNN = Pipeline([("scaler", MinMaxScaler()),
                         ("pipeline_KNN", KNeighborsClassifier())])

pipeline_LR = Pipeline([("scaler", MinMaxScaler()),
                        ("pipeline_LogisticRegression", LogisticRegression())])

pipeline_DT = Pipeline([("scaler", MinMaxScaler()),
                        ("pipeline_DecisionTree", DecisionTreeClassifier())])

pipeline_RF = Pipeline([("scaler", MinMaxScaler()),
                        ("pipeline_RandomForest", RandomForestClassifier())])

pipeline_GBC = Pipeline([("scaler", MinMaxScaler()), (
                        "pipeline_GBC", GradientBoostingClassifier())])

pipeline_SGD = Pipeline([("scaler", MinMaxScaler()),
                        ("pipeline_SGD", SGDClassifier(max_iter=5000, random_state=0))])


pipelines = [pipeline_SVM, pipeline_KNN, pipeline_LR, pipeline_DT, pipeline_RF, pipeline_GBC,
             pipeline_SGD]

pipe_dict = {0: "SupportVectorClassifier",
             1: "KNeighborsClassifier",
             2: "LogisticRegression",
             3: "DecisionTreeClassifier",
             4: "RandomForestClassifier",
             5: "GradientBoostingClassifier",
             6: "XGBClassifier",
             7: "LGBMClassifier",
             8: "SGDClassifier"}

modelNames = ["SupportVectorClassifier", "KNeighborsClassifier", "LogisticRegression", "DecisionTreeClassifier",
              "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier", "SGDClassifier"]

i = 0
trainScores = []
testScores = []

for pipe in pipelines:
    pipe.fit(X_train, y_train)
    print(f'{pipe_dict[i]}')
    print("Train Score of %s: %f     " %
          (pipe_dict[i], pipe.score(X_train, y_train)*100))
    trainScores.append(pipe.score(X_train, y_train)*100)

    print("Test Score of %s: %f      " %
          (pipe_dict[i], pipe.score(X_test, y_test)*100))
    testScores.append(pipe.score(X_test, y_test)*100)
    print(" ")

    y_predictions = pipe.predict(X_test)
    conf_matrix = confusion_matrix(y_predictions, y_test)
    print(f'Confussion Matrix: \n{conf_matrix}\n')
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]

    total = tn + fp + tp + fn
    real_positive = tp + fn
    real_negative = tn + fp

    accuracy = (tp + tn) / total  # Accuracy Rate
    precision = tp / (tp + fp)  # Positive Predictive Value
    recall = tp / (tp + fn)  # True Positive Rate
    f1score = 2 * precision * recall / (precision + recall)
    specificity = tn / (tn + fp)  # True Negative Rate
    error_rate = (fp + fn) / total  # Missclassification Rate
    prevalence = real_positive / total
    miss_rate = fn / real_positive  # False Negative Rate
    fall_out = fp / real_negative  # False Positive Rate

    print('Evaluation Metrics:')
    print(f'Accuracy    : {accuracy}')
    print(f'Precision   : {precision}')
    print(f'Recall      : {recall}')
    print(f'F1 score    : {f1score}')
    print(f'Specificity : {specificity}')
    print(f'Error Rate  : {error_rate}')
    print(f'Prevalence  : {prevalence}')
    print(f'Miss Rate   : {miss_rate}')
    print(f'Fall Out    : {fall_out}')

    print("")
    print(
        f'Classification Report: \n{classification_report(y_predictions, y_test)}\n')
    print("")

    print("*****"*20)
    i += 1
