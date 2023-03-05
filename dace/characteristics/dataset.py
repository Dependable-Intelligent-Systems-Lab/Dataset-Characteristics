"""
The main module for getting the data characteristics.
"""
import pandas
import skdim
import numpy as np
from pingouin import multivariate_normality
from sklearn.metrics.cluster import homogeneity_score
import pingouin

def getDatasetDimensionality(datasetInDataFrameFormat: pandas.DataFrame) -> int:
    """To get the dimensionality (number of features/attributes present in a dataset) of a dataset passed as dataframe in pandas format.
    Note: Class/Label colum is not considers as a feature here

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset

    Returns:
        int: dimentionality-number of features present in the dataset
    """
    #number of columns ignoring the class column
    return datasetInDataFrameFormat.shape[1] - 1


def getDatasetNumberOfInstances(datasetInDataFrameFormat:  pandas.DataFrame) -> int:
    """To get the number of instances present in a dataset passed as dataframe in pandas format.

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset

    Returns:
        int: number of instances present in a dataset
    """
    return datasetInDataFrameFormat.shape[0]

def getDatasetNumberOfClasses(datasetInDataFrameFormat: pandas.DataFrame, className: str) -> int:
    """To get the number of classes present in a dataset for a class.

    Args:
        datasetInDataFrameFormat (pandas.DataFrame): dataset
        className (str): colum name where classes/Labels are present

    Returns:
        int: number of classes present in a dataset for the given Label
    """
    return len(datasetInDataFrameFormat[className].unique())

def getNumberOfZerosInDataset(df_Dataset: pandas.DataFrame) -> int:
    """To get the number of zeros present in a dataset

    Args:
        df_Dataset (pandas.DataFrame): dataset

    Returns:
        int: number of zeros present in a dataset for all the features
    """
    return (df_Dataset==0).sum().sum()

def getNumberOfNaNsInDataset(df_Dataset: pandas.DataFrame) -> int:
    """To get the number of NaN (null values) present in a dataset

    Args:
        df_Dataset (pandas.DataFrame): dataset

    Returns:
        int: number of NaNs present in a dataset for all the features
    """
    return df_Dataset.isnull().sum().sum()

def getNumberOfZerosInAdatasetFeature(df_Dataset: pandas.DataFrame, featureName: str) -> int:
    """To get the number of zeros present in a dataset for a given feature column

    Args:
        df_Dataset (pandas.DataFrame): dataset
        featureName (str): column name of the feature

    Returns:
        int: number of zeros present in a dataset for a given feature
    """
    arr = (df_Dataset==0).sum()[featureName]
    return arr

def getNumberOfNaNsInAdatasetFeature(df_Dataset: pandas.DataFrame, featureName: str) -> int:
    """To get the number of NaNs present in a dataset for a given feature column

    Args:
        df_Dataset (pandas.DataFrame): dataset
        featureName (str): column name of the feature

    Returns:
        int: number of NaNs present in a dataset for a given feature
    """
    return df_Dataset[featureName].isnull().sum()

def getZeroSparsity(df_Dataset: pandas.DataFrame) -> float:
    """To get the zero sparsity measure. number of zeros present in a dataset divided by total number of entries present in a dataset

    Args:
        df_Dataset (pandas.DataFrame): dataset

    Returns:
        float: zero sparsity in fraction and not in percentage
    """
    return getNumberOfZerosInDataset(df_Dataset)/df_Dataset.count().sum()

def getNaNSparsity(df_Dataset: pandas.DataFrame) -> float:
    """To get the NaN sparsity measure. number of NaNs present in a dataset divided by total number of entries present in a dataset

    Args:
        df_Dataset (pandas.DataFrame): dataset in pandas dataframe format

    Returns:
        float: NaN sparsity in fraction and not in percentage
    """
    return getNumberOfNaNsInDataset(df_Dataset)/df_Dataset.count().sum()

def getDataSparsity(df_Dataset: pandas.DataFrame, className: str) -> float:
    """To get the data sparsity as per the published paper. DataSparsity = pow(N,1/d)

    Args:
        df_Dataset (pandas.DataFrame): dataset
        className (str): name of the class column in the dataset

    Returns:
        float: calculated data sparsity
    """
    N = getDatasetNumberOfClasses(df_Dataset, className)
    d = getDatasetDimensionality(df_Dataset)
    #print(N, d)
    return N ** (1/d)

def getFeatureNoise(df_DatasetFeatures: pandas.DataFrame) -> float:
    """To calculate the feature noise (feature noise here is defined as average of all the variance of feature columns)

    Args:
        df_DatasetFeatures (pandas.DataFrame): dataset features

    Returns:
        float: feature noise (average of variances of features)
    """
    sumVariance = 0
    count = df_DatasetFeatures.shape[1]
    for (colName, colData) in df_DatasetFeatures.iteritems():
        sumVariance = sumVariance + colData.var()
        #print('Column Name : ', colData)
        #print('Column Contents : ', colData.values)
    return sumVariance/count

def getIntrinsicDimensionaltiy(df_DatasetFeatures: pandas.DataFrame) -> int:
    """Function to calculate the Intrinsic Dimensionality (ID) based on PCA.

    Args:
        df_DatasetFeatures (pandas.DataFrame): dataset features

    Returns:
        int: intrinsic dimensionality
    """
    pca = skdim.id.lPCA().fit(df_DatasetFeatures)
    return pca.dimension_

def getIntrinsicDimensionaltiyRatio(df_DatasetFeatures: pandas.DataFrame) -> float:
    """Function to calculate the Intrinsic Dimensionality (ID) based on PCA

    Args:
        df_DatasetFeatures (pandas.DataFrame): dataset features

    Returns:
        float: intrinsic dimensionality ratio
    """
    return getIntrinsicDimensionaltiy(df_DatasetFeatures)/(getDatasetDimensionality(df_DatasetFeatures)+1)

def getFeatureNoise2(df_DatasetFeatures: pandas.DataFrame) -> float:
    """To calculate the feature noise (as per the translated paper = (d-ID)/ID; d=dimensionality, ID=Intrinsic dimensionality)

    Args:
        df_DatasetFeatures (pandas.DataFrame): dataset features

    Returns:
        float: feature noise (as per the translated paper, (d-ID)/ID)
    """
    d = getDatasetDimensionality(df_DatasetFeatures)+1
    ID = getIntrinsicDimensionaltiy(df_DatasetFeatures)
    return (d-ID)/d

def getCorrelationOfFeaturesWithNoClass(df_Dataset: pandas.DataFrame, className: str) -> float:
    """To get the aveage/mean of absolute correlation value for all the features without considering the class.

    Args:
        df_Dataset (pandas.DataFrame): dataset
        className (str): name of the class/label column in the dataset

    Returns:
        float: calculated correlation coefficient for the features without considering the class
    """
    df_DatasetFeatures = df_Dataset.drop(className, axis=1)#removing the class/label column
    df_corr = df_DatasetFeatures.corr().abs() #calculating the absolute value of pearson correlation
    df_corr = df_corr.replace(1.0, np.nan) #replacing the correlation between the same features to NaN to ignore for mean calculation
    #print(df_var)
    return df_corr.mean().mean() #mean of all the absolute values of correlation of features

def getCorrelationOfFeaturesWithClass(df_Dataset: pandas.DataFrame, className: str) -> float:
    """To get the aveage/mean of absolute correlation value for all the features considering the class (as per the paper). Correlation of features for each class is calculated seperatly and mean of the same is taken here.

    Args:
        df_Dataset (pandas.DataFrame): dataset
        className (str): name of the class/label column in the dataset

    Returns:
        float: calculated correlation coefficient for the features considering the class
    """
    lstLabels = df_Dataset[className].unique() #get all the labels
    count = 0
    sum = 0
    #calculate the correlation of features for each class seperatly
    for label in lstLabels:
        df_DatasetLabel = df_Dataset.loc[df_Dataset[className] == label]
        #print(df_DatasetLabel)
        sum = sum + getCorrelationOfFeaturesWithNoClass(df_DatasetLabel, className)
        count = count + 1;
    return sum/count #average/mean of correlation calculated for each class

def getMultiVariateNormality(df_DatasetFeatures: pandas.DataFrame) -> pingouin.multivariate.HZResults:
    """To calculate the multivariate normality using Henze-Zirkler multivariate normality test

    Args:
        df_DatasetFeatures (pandas.DataFrame): dataset features

    Returns:
        pingouin.multivariate.HZResults: multivariate normality test results
    """
    return multivariate_normality(df_DatasetFeatures, alpha=.05)

def isMultiVariateNormalityExists(df_DatasetFeatures: pandas.DataFrame) -> bool:
    """To calculate the multivariate normality using Henze-Zirkler multivariate normality test and returns true if MVN exists else returns false

    Args:
        df_DatasetFeatures (pandas.DataFrame): dataset features

    Returns:
        bool: returns True if multivariate normality exists
    """
    MVN = getMultiVariateNormality(df_DatasetFeatures)
    return MVN.normal

def getAllCovarianceOfFeatures(df_Dataset: pandas.DataFrame, className: str) -> list:
    """To get the list containing all the covariances between features for calculating the homogenity of class covariances.

    Args:
        df_Dataset (pandas.DataFrame): dataset
        className (str): name of the class/label column in the dataset

    Returns:
        list: list containing all the calculated covariances (float) between features
    """
    lstCovars = []
    df_DatasetFeatures = df_Dataset.drop(className, axis=1)#removing the class/label column
    df_var = df_DatasetFeatures.cov()#.abs() #calculating the covariances of features
    lst_vars = df_var.values.tolist()# converting to list
    #converting to a single list
    for lst_var in lst_vars:
        for val in lst_var:
            lstCovars.append(val)
    df_Covars = pandas.DataFrame(lstCovars, columns=['FeatureCovariances'])# converting to a dataframe for easy unique() value findings
    return (df_Covars['FeatureCovariances'].unique().tolist()) #only unique values are retained in the list

def getHomogeneityOfClassCovariance(df_Dataset: pandas.DataFrame, className: str) -> float:
    """To get the average of homogeneity of Covariances between features of each classes.

    Args:
        df_Dataset (pandas.DataFrame):  dataset
        className (str): name of the class/label column in the dataset

    Returns:
        float: Homogeneity of class covariances
    """
    lstLabels = df_Dataset[className].unique()
    lstClassCovariances = []
    sumHomogeneity = 0
    count = 0
    #storing the variances for classes in a list
    for label in lstLabels:
        #X_train_Class = df_X_train.loc[df_Y_train == label]
        lstClassCovariances.append(getAllCovarianceOfFeatures(df_Dataset.loc[df_Dataset[className]== label], className))
    #calculating the homogeneity
    for i in range(len(lstClassCovariances)):
        for j in range(i+1,len(lstClassCovariances)):
            count = count + 1
            try:
                sumHomogeneity = sumHomogeneity + homogeneity_score(lstClassCovariances[i], lstClassCovariances[j])
            except:
                sumHomogeneity = 0
            #print ([i,j])
            #print(lstClassCovariances[i])
            #print(lstClassCovariances[j])
            
    return sumHomogeneity/count #taking the average of all the homogenities of classes

def getAllDatasetCharacteristicsTable(df_Dataset: pandas.DataFrame, className: str) -> pandas.DataFrame:
    """To get all the dataset characteristics calculated and populated in the output

    Args:
        df_Dataset (pandas.DataFrame): dataset
        className (str): name of the class/label column in the dataset

    Returns:
        pandas.DataFrame: All dataset characteristics 
    """
    df_FeaturesWithoutClass = df_Dataset.drop(className, axis=1)
    lst_ColNames = ['Parameters', 'Value']
    df_table = pandas.DataFrame(columns = lst_ColNames)#creating empty dataframe
    
    #dimenstionality calculation and appending to the dataframe
    d = getDatasetDimensionality(df_Dataset)
    df_table = df_table.append({'Parameters' : 'Dimensionality (d)', 'Value':d}, 
                ignore_index = True)
    
    #Number of instances calculation
    N = getDatasetNumberOfInstances(df_Dataset)
    df_table = df_table.append({'Parameters' : 'NrOfInstances (N)', 'Value':N}, 
                ignore_index = True)
    
    #Number of classes calculation
    C = getDatasetNumberOfClasses(df_Dataset, className)
    df_table = df_table.append({'Parameters' : 'NrOfClasses (C)', 'Value':C}, 
                ignore_index = True)
    
    #Zero data sparsity calculation - ignoring classes ?
    OS = getZeroSparsity(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'ZeroSparsity (OS)', 'Value' : OS}, 
                ignore_index = True)
    
    #NaN data sparsity calculation - ignoring classes?
    NS = getNaNSparsity(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'NaNSparsity (NS)', 'Value':NS}, 
                ignore_index = True)
    
    #Data sparsity calculation
    DS = getDataSparsity(df_Dataset, className)
    df_table = df_table.append({'Parameters' : 'DataSparsity (DS)', 'Value':DS}, 
                ignore_index = True)
    
    #Data sparsity ratio - TBD-needed?
    df_table = df_table.append({'Parameters' : 'DataSparsityRatio (DSR)', 'Value':'TBD'}, 
                ignore_index = True)
    
    #Pearson correlation value of features considering classes
    CorrFC = getCorrelationOfFeaturesWithClass(df_Dataset, className)
    df_table = df_table.append({'Parameters' : 'Correlation of Featues with Class (CorrFC)', 'Value':CorrFC}, 
                ignore_index = True)
    
    #Pearson correlation value of features without considering classes
    CorrFNC = getCorrelationOfFeaturesWithNoClass(df_Dataset, className)
    df_table = df_table.append({'Parameters' : 'Correlation of Featues without Class (CorrFNC)', 'Value':CorrFNC}, 
                ignore_index = True)
    
    #Multivariate normality calculation
    MVN = isMultiVariateNormalityExists(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'Multivariate normality? (MVN)', 'Value': MVN}, 
                ignore_index = True)
    
    #Homogeneity of class covariances calculation
    HCCov = getHomogeneityOfClassCovariance(df_Dataset, className)
    df_table = df_table.append({'Parameters' : 'Homogeneity of class covariance (HCCov)', 'Value':HCCov}, 
                ignore_index = True)
    
    #Intrinsic dimensionality calculation
    ID = getIntrinsicDimensionaltiy(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'Intrinsic Dimensionality-PCA (ID)', 'Value':ID}, 
                ignore_index = True)
    
    #Intrinsic dimensionality ratio
    IDR = getIntrinsicDimensionaltiyRatio(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'Intrinsic Dimensionality Ratio (IDR)', 'Value':IDR}, 
                ignore_index = True)
    
    #Feature noise1 calculation as per variance
    FN1 = getFeatureNoise(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'Feature Noise variance (FN1)', 'Value':FN1}, 
                ignore_index = True)
    
    #Feature noise2 calculation as per paper
    FN2 = getFeatureNoise2(df_FeaturesWithoutClass)
    df_table = df_table.append({'Parameters' : 'Feature Noise paper (FN2)', 'Value':FN2}, 
                ignore_index = True)
    
    return df_table