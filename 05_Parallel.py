import pandas as pd
import numpy as np
from keras.models import model_from_json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time


# Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

##################################################################################################
# ************************* Regression Model Evaluation ***************************************
##################################################################################################

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


debug = __import__('04_Debug')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)


combined_dataset = data_prep.get_combined_dataset()
ds_train, ds_valid, ds_test = data_prep.split_data_regression(
    combined_dataset)
ds_train = ds_train.append(ds_valid)
X_train, X_test, y_train, y_test = data_prep.load_data(ds_train, ds_test)


def Shap_processing_Regression():
    # Place data into DataFrame for readability
    X_test_frame = pd.DataFrame(X_test)
    X_test_frame.columns = ['fixed acidity', 'volatile acidity', 'citric acid',
                            'residual sugar', 'chlorides', 'free sulfur dioxide',
                            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                            'red', 'white']

    X_train_frame = pd.DataFrame(X_train)
    X_train_frame.columns = ['fixed acidity', 'volatile acidity', 'citric acid',
                             'residual sugar', 'chlorides', 'free sulfur dioxide',
                             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                             'red', 'white']

    # Summarize the training set to accelerate analysis
    X_train_frame = shap.kmeans(X_train_frame.values, 25)
    loaded_model = Tuning.Regression_Tuning()
    # Instantiate an explainer with the model predictions and training data (or training data summary)
    explainer = shap.KernelExplainer(loaded_model.predict, X_train_frame)


# red wine visualization

    # Extract Shapley values from the explainer
    # Select test data representing red wine category
    shap_values = explainer.shap_values(
        X_test_frame[X_test_frame['red'] == 1][:400])

    # Summarize the Shapley values in a plot
    plt.title('Feature impact on model output')
    shap.summary_plot(
        shap_values[0][:, :-2], X_test_frame[X_test_frame['red'] == 1][:400].iloc[:, :-2][:400])

    # Plot the SHAP values for one red wine sample
    INSTANCE_NUM = 50

    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][INSTANCE_NUM, :-2],
                    X_test_frame[X_test_frame['red'] == 1].iloc[INSTANCE_NUM, :-2])

    # Plot the SHAP values for multiple red wine samples

    INSTANCE_NUM = list(np.arange(100))

    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][INSTANCE_NUM, :-2],
                    X_test_frame[X_test_frame['red'] == 1].iloc[INSTANCE_NUM, :-2])


# white wine

    # Extract Shapley values from the explainer
    # Select test data representing white wine category
    shap_values = explainer.shap_values(
        X_test_frame[X_test_frame['white'] == 1][:400])

    # Summarize the Shapley values in a plot
    plt.title('Feature impact on model output')
    shap.summary_plot(
        shap_values[0][:, :-2], X_test_frame[X_test_frame['white'] == 1][:400].iloc[:, :-2][:400])

    # Plot the SHAP values for one white wine sample
    INSTANCE_NUM = 42

    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][INSTANCE_NUM, :-2],
                    X_test_frame[X_test_frame['white'] == 1].iloc[INSTANCE_NUM, :-2])

    # Plot the SHAP values for multiple white wine samples

    INSTANCE_NUM = list(np.arange(100))

    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][INSTANCE_NUM, :-2],
                    X_test_frame[X_test_frame['white'] == 1].iloc[INSTANCE_NUM, :-2])


def EDA():
    data_prep.evaluating_dataset()


def Sequential():
    start_Seq = time.time()
    start_Seq_classification = time.time()

    debug.Evaluate_Classification_Model()
    end_Seq_classification = time.time()

    start_Seq_regression = time.time()

    debug.Evaluate_Regression_Model()
    end_Seq_regression = time.time()
    end_Seq = time.time()

    Sequential_Time = end_Seq-start_Seq
    Sequential_Time_Classification = end_Seq_classification - start_Seq_classification
    Sequential_Time_Regression = end_Seq_regression - start_Seq_regression

    print("Sequential Execution time", Sequential_Time)
    print("Sequential Execution time Classification",
          Sequential_Time_Classification)
    print("Sequential Execution time Regression", Sequential_Time_Regression)


Sequential()
