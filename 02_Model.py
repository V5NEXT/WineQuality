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


# model callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=0)  # val_loss


callbacks = [early_stopping]

# # Using a Dummy Classifer for checking scores
# print(X_test)
# print(y_test)

# dummy_classifier = DummyClassifier(strategy='most_frequent', random_state=2020)
# dummy_classifier.fit(X_train, y_train)
# acc_baseline = dummy_classifier.score(X_test, y_test)
# print("Baseline Accuracy = ", acc_baseline)


# # model1 : Support Vector Classifier
# def RegressionModel1():
#     from sklearn.svm import SVC
#     svc = SVC(random_state=2020)
#     svc.fit(X_train, y_train)

#     from sklearn import metrics
#     from sklearn.metrics import accuracy_score
#     y_pred = svc.predict(X_test)
#     print(metrics.accuracy_score(y_test, y_pred))

#     # prevent overfitting

#     from sklearn.model_selection import cross_val_score
#     X, y = data_prep.basic_preprocessing()
#     scores = cross_val_score(svc, X, y, cv=5)
#     print(scores.mean())

#     y_pred_train = svc.predict(X_train)
#     print(metrics.accuracy_score(y_train, y_pred_train))

#     # regularixation using randomizedSearch CV
#     # from sklearn.model_selection import RandomizedSearchCV
#     # random_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#     # svc_random = RandomizedSearchCV(svc, random_grid, cv=5, random_state=2020)
#     # svc_random.fit(X_train, y_train)
#     # print(svc_random.best_params_)

#     # using gridsearchCV
#     # from sklearn.model_selection import GridSearchCV
#     # param_dist = {'C': [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
#     #               'kernel': ['linear', 'rbf', 'poly']}
#     # svc_cv = GridSearchCV(svc, param_dist, cv=10)
#     # svc_cv.fit(X_train, y_train)
#     # print(svc_cv.best_params_)

#     # final SVM model (after finding best params)
#     # svc_new = SVC(C=1.3, kernel="rbf", random_state=2020)
#     # svc_new.fit(X_train, y_train)
#     # y_pred_new = svc_new.predict(X_test)
#     # print(metrics.accuracy_score(y_test, y_pred_new))


# # Decisson Tree
# def RegressionModel2():
#     from sklearn.tree import DecisionTreeClassifier
#     dt = DecisionTreeClassifier(random_state=2020)
#     dt.fit(X_train, y_train)

#     from sklearn.metrics import plot_confusion_matrix
#     y_pred = dt.predict(X_test)
#     metrics.plot_confusion_matrix(dt, X_test, y_test)
#     plt.show()
#     print(metrics.accuracy_score(y_test, y_pred))


# # Random Forest

# def RegressionModel3():
#     from sklearn.ensemble import RandomForestClassifier
#     rf_model = RandomForestClassifier(random_state=2020)
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
#     acc_rf = accuracy_score(y_test, y_pred_rf)
#     print('Accuracy = ', acc_rf)


# # RegressionModel1()  # 81.307
# # RegressionModel2()  # 82.538
# # RegressionModel3()  # 87.846


# # Since Classification Model3 (Random Forest Classfier yielded the best results I am considering it as the main Model)


# def RegressionFinalModel():
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import cross_val_score
#     from sklearn.model_selection import RandomizedSearchCV

#     rf_model = RandomForestClassifier(random_state=2020)
#     rf_model.fit(X_train, y_train)
#     y_pred_rf = rf_model.predict(X_test)
#     acc_rf = accuracy_score(y_test, y_pred_rf)
#     print('Accuracy = ', acc_rf)

#     # prevent overfitting
#     X, y = data_prep.basic_preprocessing()

#     scores = cross_val_score(rf_model, X, y, cv=5)
#     print("Cross Validation Score: ", scores.mean())

#     y_pred_train = rf_model.predict(X_train)
#     print(metrics.accuracy_score(y_train, y_pred_train))


# # Even though random forest has corrected for decision tree’s habit of overfitting (to some extent),
# #  the disparity between cross validation score and training accuracy here indicates that our random forest model is still overfitting a bit.
# #  Similar to decision tree, we can prune some hyperparameters such as max-depth and n_estimators by using GridSearchCV to address overfitting.
#     print(rf_model.get_params())

# # fine tuning by changing the number of estimators and depth

#     random_grid = {'max_depth': [1, 5, 10, 15],
#                    'n_estimators': [100, 200, 300, 400, 500, 600]}
#     rf_random = RandomizedSearchCV(
#         rf_model, random_grid, n_iter=50, cv=5, random_state=2020)
#     rf_random.fit(X_train, y_train)
#     print(rf_random.best_params_)
#     rf_new = RandomForestClassifier(
#         n_estimators=450, max_depth=14, random_state=2020)
#     rf_new.fit(X_train, y_train)
#     y_pred_rf = rf_new.predict(X_test)
#     acc_rf = accuracy_score(y_test, y_pred_rf)
#     print('Accuracy = ', acc_rf)
#     scores = cross_val_score(rf_new, X, y, cv=5)
#     print("Cross Validation Score: ", scores.mean())


# # RegressionFinalModel()


# Prediction of Wine Type


def ClassificationModel():
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

    plotValLossAndLoss(history)

    # Predicting the Value
    y_pred = model.predict(X_test)
    print(y_pred)

    filename = "finalized_model_classification.joblib"
    joblib.dump(model, filename)


ClassificationModel()


# Regression Model
