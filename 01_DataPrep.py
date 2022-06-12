import os
import glob
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


os.chdir("Data_Set")

combined_df = pd.DataFrame
new_df = pd.DataFrame


def get_combined_dataset():
    whitewine_df = pd.read_csv('winequality-white.csv', delimiter=';')
    redwine_df = pd.read_csv('winequality-red.csv', delimiter=';')

    whitewine_df['wine_type'] = 0
    redwine_df['wine_type'] = 1
    combined_df = pd.concat([whitewine_df, redwine_df])
    return combined_df


def evaluating_dataset():
    combined_df = get_combined_dataset()
    print(combined_df.info())
    print(combined_df.describe())
    combined_df.hist(bins=25, figsize=(10, 10))
    # display histogram
    plt.show()

    plt.figure(figsize=[10, 6])
    # plot bar graph
    plt.bar(combined_df['quality'], combined_df['alcohol'], color='red')
    # label x-axis
    plt.xlabel('quality')
    # label y-axis
    plt.ylabel('alcohol')

    plt.show()

    # ploting heatmap (for correlation)
    plt.figure(figsize=[19, 10], facecolor='blue')
    sb.heatmap(combined_df.corr(), annot=True)
    plt.show()


def basic_preprocessing():
    combined_df = get_combined_dataset()

    for a in range(len(combined_df.corr().columns)):
        for b in range(a):
            if abs(combined_df.corr().iloc[a, b]) > 0.7:
                name = combined_df.corr().columns[a]
                print(name)

    # dropping total sulfur dioxide coloumn for low correlation as to reduce features
    new_df = combined_df.drop('total sulfur dioxide', axis=1)

    # In the dataset, there is so much notice data present, which will affect the accuracy of our ML model

    new_df.isnull().sum()

    # We see that there are not many null values are present in our data so we simply fill them with the help of the fillna() function

    new_df.update(new_df.fillna(new_df.mean()))
    scaler = MinMaxScaler(feature_range=(0, 1))

    normal_df = scaler.fit_transform(new_df)
    normal_df = pd.DataFrame(normal_df, columns=new_df.columns)
    print(normal_df.head())

    new_df["good wine"] = ["yes" if i >=
                           7 else "no" for i in new_df['quality']]

    X = normal_df.drop(["quality"], axis=1)
    y = new_df["good wine"]

    y.value_counts()
    sns.countplot(y)
    plt.show()

    return X, y


def split_dataset():
    train, test = basic_preprocessing()

    # set aside 20% of train and test data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(train, test,
                                                        test_size=0.2, shuffle=True, random_state=8)
    # Use the same function above for the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.25, random_state=8)  # 0.25 x 0.8 = 0.2

    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_train shape: {}".format(X_val.shape))
    print("y_test shape: {}".format(y_test.shape))
    print("X_val shape: {}".format(y_train.shape))
    print("y val shape: {}".format(y_val.shape))

    print("##################### Length #####################")
    print(f'Total # of sample in whole dataset: {len(X_train)+len(X_test)}')
    print(f'Total # of sample in train dataset: {len(X_train)}')
    print(f'Total # of sample in test dataset: {len(X_test)}')

    print("##################### Shape #####################")
    print(f'Shape of train dataset: {X_train.shape}')
    print(f'Shape of test dataset: {X_test.shape}')

    print("##################### Percantage #####################")
    print(
        f'Percentage of train dataset: {round((len(X_train)/(len(X_train)+len(X_test)))*100,2)}%')
    print(
        f'Percentage of validation dataset: {round((len(X_test)/(len(X_train)+len(X_test)))*100,2)}%')

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset_classification():
    combined_df = get_combined_dataset()

    X = combined_df.iloc[:, 0:11]
    y = np.ravel(combined_df.wine_type)

    # Splitting the data set for training and validating
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=45)

    # Use the same function above for the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.25, random_state=8)  # 0.25 x 0.8 = 0.2

    return X_train, X_test, y_train, y_test, X_val, y_val
# split_dataset()


# split_dataset_classification()
