from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.metrics
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


# Using a Dummy Classifer for checking scores
print(X_test)
print(y_test)

dummy_classifier = DummyClassifier(strategy='most_frequent', random_state=2020)
dummy_classifier.fit(X_train, y_train)
acc_baseline = dummy_classifier.score(X_test, y_test)
print("Baseline Accuracy = ", acc_baseline)


# model1 : Support Vector Classifier
def Classification_Model1():
    from sklearn.svm import SVC
    svc = SVC(random_state=2020)
    svc.fit(X_train, y_train)

    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    y_pred = svc.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))

    # prevent overfitting

    from sklearn.model_selection import cross_val_score
    X, y = data_prep.basic_preprocessing()
    scores = cross_val_score(svc, X, y, cv=5)
    print(scores.mean())

    y_pred_train = svc.predict(X_train)
    print(metrics.accuracy_score(y_train, y_pred_train))

    # regularixation using randomizedSearch CV
    from sklearn.model_selection import RandomizedSearchCV
    random_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    svc_random = RandomizedSearchCV(svc, random_grid, cv=5, random_state=2020)
    svc_random.fit(X_train, y_train)
    print(svc_random.best_params_)

    # using gridsearchCV
    from sklearn.model_selection import GridSearchCV
    param_dist = {'C': [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
                  'kernel': ['linear', 'rbf', 'poly']}
    svc_cv = GridSearchCV(svc, param_dist, cv=10)
    svc_cv.fit(X_train, y_train)
    print(svc_cv.best_params_)

    # final SVM model (after finding best params)
    svc_new = SVC(C=1.3, kernel="rbf", random_state=2020)
    svc_new.fit(X_train, y_train)
    y_pred_new = svc_new.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred_new))


# Classification_Model1()


# Decisson Tree
def ClassificationModel2():
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=2020)
    dt.fit(X_train, y_train)

    from sklearn.metrics import plot_confusion_matrix
    y_pred = dt.predict(X_test)
    metrics.plot_confusion_matrix(dt, X_test, y_test)
    plt.show()
    print(metrics.accuracy_score(y_test, y_pred))
