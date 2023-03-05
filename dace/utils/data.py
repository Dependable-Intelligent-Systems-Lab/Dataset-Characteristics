"""
This module contains the functions for data manipulation.
"""

import pandas
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn


def readDataset2DataFrame(datasetFullPathInCSVformat: str) -> pandas.DataFrame:
    """To read the dataset in .csv format and return the dataset in pandas dataframe

    Args:
        datasetFullPathInCSVformat (str): full path of the dataset to be read in csv format

    Returns:
        pandas.DataFrame: dataset
    """
    return pandas.read_csv(datasetFullPathInCSVformat)

def fixDatasetColumnName(datasetInDataFrameFormat: pandas.DataFrame) -> pandas.DataFrame:
    """To fix the dataset colum name in the dataframe. It strips the spaces in the begining and end, replaces the spaces with '_', 
    converts the upper case to lower cases and replaces the braces. Doing so will help for searching in dataframe.

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset

    Returns:
        pandas.DataFrame: dataset with colum name corrected
    """
    datasetInDataFrameFormat.columns = datasetInDataFrameFormat.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return datasetInDataFrameFormat

def reduceDimensionsUsingPCA(datasetInDataFrameFormat: pandas.DataFrame, NumOfPCAcomponents: int) -> pandas.DataFrame:
    """To reduce the dimensionality of a dataset using PCA-Principal Component Analysis method.

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset for which dimensions to be reduced
        NumOfPCAcomponents (int): num of PCA components in a reduced dataset

    Returns:
        pandas.DataFrame: reduced dimensions dataset
    """
    pca = PCA(n_components = NumOfPCAcomponents)
    pca.fit(datasetInDataFrameFormat)
    #plot the variance graph
    plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    #transform the pca to dataframe
    df = pca.transform(datasetInDataFrameFormat)
    #setting the column names
    colNames = []
    for i in range(NumOfPCAcomponents):    
        colNames.append(f"Principal_component_{i+1}")
    df = pandas.DataFrame(data = df, columns = colNames)
    return df

def scaleDatasetUsingStandardScalar(datasetInDataFrameFormat: pandas.DataFrame) -> pandas.DataFrame:
    """To scale the features into unit scale for faster processing.

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset to be scaled

    Returns:
        pandas.DataFrame: scaled dataset
    """
    columnNames = datasetInDataFrameFormat.columns
    df= StandardScaler().fit_transform(datasetInDataFrameFormat)
    df = pandas.DataFrame(data = df, columns = columnNames)
    return df

def getSplitTrainNtestDataUsingStratKfold(InputX: pandas.DataFrame, OutputY: pandas.DataFrame, folds: int = 10) -> list:
    """To get the split data for training and testing using the stratified K fold splitting approach.

    Args:
        InputX (pandas.DataFrame): independant input variables for splitting
        OutputY (pandas.DataFrame): dependant output variables for splitting
        folds (int, optional): number of folds for stratified K fold splitting. Defaults to 10.

    Returns:
        list: [X_train, X_test, y_train, y_test] split data for training and testing
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    #Below loop, goes through all the 10 splits and takes the last split data only
    for train_index, test_index in skf.split(InputX, OutputY):
        X_train, X_test = InputX.iloc[train_index], InputX.iloc[test_index]
        y_train, y_test = OutputY.iloc[train_index], OutputY.iloc[test_index]
    return [X_train, X_test, y_train, y_test]

def getRFpossibleAccuraciesWithStatSplit(InputX: pandas.DataFrame, OutputY: pandas.DataFrame, numberOfSplits: int = 10, estimators: int = 200) -> list:
    """To get the possible accuracies for the RF model for a n stratified K fold splits

    Args:
        InputX (pandas.DataFrame): independant input variables for splitting
        OutputY (pandas.DataFrame):  dependant output variables for splitting
        numberOfSplits (int, optional): number of folds for stratified K fold splitting. Defaults to 10.
        estimators (int, optional): number of estimators for RF model. Defaults to 200.

    Returns:
        list: accuracies for n splits of data
    """
    Accuracies = []
    RFmodel = RandomForestClassifier(n_estimators = estimators, criterion = 'entropy', random_state = 0)
    skf = StratifiedKFold(n_splits=numberOfSplits, shuffle=True, random_state=1)
    #Below loop, goes through all the 10 splits and takes the last split data only
    for train_index, test_index in skf.split(InputX, OutputY):
        X_train, X_test = InputX.iloc[train_index], InputX.iloc[test_index]
        y_train, y_test = OutputY.iloc[train_index], OutputY.iloc[test_index]
        RFmodel.fit(X_train,y_train)
        predicted_y = RFmodel.predict(X_test)
        Accuracies.append(accuracy_score(y_test, predicted_y)*100)#accuracy in percentage
    return Accuracies

def getRandomForestModelForDataset(X_train: pandas.DataFrame, Y_train: pandas.DataFrame, estimators: int =200) -> sklearn.ensemble.RandomForestClassifier:
    """_summary_

    Args:
        X_train (pandas.DataFrame): independant variables for training
        Y_train (pandas.DataFrame): dependant variables for training
        estimators (int, optional): number of estimators for random forest. Defaults to 200.

    Returns:
        sklearn.ensemble.RandomForestClassifier: random forest model generated using the trained data
    """
    RFmodel = RandomForestClassifier(n_estimators = estimators, criterion = 'entropy', random_state = 0)
    RFmodel.fit(X_train,Y_train)
    return RFmodel

def getPredictedValuesNaccuracy(model: sklearn.ensemble.RandomForestClassifier, X_Test: pandas.DataFrame, expected_y: pandas.DataFrame) -> list([list, float]):
    """To generate the predicted values and accuracy for a model using test inputs and expected outputs.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): _description_
        X_Test (pandas.DataFrame): dependant variables for testing 
        expected_y (pandas.DataFrame): expected output for a test dataset

    Returns:
        list([list, flaot]): predicted outputs and accuracy of the model for the predicted outputs
    """
    predicted_y = model.predict(X_Test)
    return [predicted_y, accuracy_score(expected_y, predicted_y)*100]#accuracy in percentage

