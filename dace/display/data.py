"""
This module contains the functions for display of miscellaneous data.
"""

import pandas as pd
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def displayConfusionMatrixHeatMap(actualValuesXaxis: pandas.core.series.Series, predictedValuesYaxis: pandas.core.series.Series):
    """To plot the confustion matrix as heatmap with the actual and predicted values

    Args:
        actualValuesXaxis (pandas.core.series.Series):  actual values/labels to plot on X axis
        predictedValuesYaxis (pandas.core.series.Series): predicted values/labels to plot on Y axis
    """
    cm_matrix = confusion_matrix(actualValuesXaxis, predictedValuesYaxis)
    # Displaying dataframe, cm_matrix as an heatmap with diverging colourmap as RdYlGn
    sns.heatmap(cm_matrix, cmap ='RdYlGn', linewidths = 0.30, annot = True)
    plt.xlabel('Actual Label')
    plt.ylabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

def displayClassificationReport(actualValuesXaxis: pandas.core.series.Series, predictedValuesYaxis: pandas.core.series.Series) -> pandas.DataFrame:
    """To display the classification report for the actual and predicted values/labels

    Args:
        actualValuesXaxis (pandas.core.series.Series): actual values/labels
        predictedValuesYaxis (pandas.core.series.Series): predicted values/labels

    Returns:
        pandas.DataFrame: classification report
    """
    report = classification_report(actualValuesXaxis, predictedValuesYaxis, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.sort_values(by=['f1-score'], ascending=False)
    return df

def displayCorrelationMatrixHeatMap(datasetInDataFrameFormat: pandas.DataFrame):
    """To plot the correlation matrix as heatmap for the features in a dataset
    This helps to identify the correlation between each features.

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset in pandas dataframe format
    """
    # Defining figure size for the output plot in inches
    fig, ax = plt.subplots(figsize = (12, 7))
    df = pd.DataFrame(datasetInDataFrameFormat, columns =datasetInDataFrameFormat.columns)
    corr = df.corr()
    plt.title('Correlation Matrix for the features')
    sns.heatmap(corr, annot = True)


def displayMultivariateAnalysisPlot(datasetInDataFrameFormat: pandas.DataFrame, classColName: str, NumberOfColumns: int):
    """To display pair plot for the multivariate analysis of features and classes 

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset
        classColName (str): colum name where output/classes/labels are present
        NumberOfColumns (int): number of features in a dataset
    """
    sns.pairplot(datasetInDataFrameFormat, hue=classColName, height=NumberOfColumns)
    plt.show()