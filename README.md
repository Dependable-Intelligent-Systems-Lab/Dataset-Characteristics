# D-ACE: Dataset Assessment and Characteristics Evaluation

<p align = 'justified'>Dataset quality assessment is a crucial aspect of machine learning and artificial intelligence, as the performance and accuracy of algorithms are directly dependent on the quality and characteristics of the data they are trained on. Poor quality datasets can lead to biased or inaccurate results, leading to incorrect decisions being made. Hence, it is important to measure the quality of datasets and identify any potential issues before using them for training machine learning models.</p>
<div style="text-align: justify;">D-ACE is a framework designed to assess the quality and characteristics of datasets, helping to identify any potential issues that may affect the performance of machine learning algorithms. This framework provides a comprehensive evaluation of the dataset, taking into account factors such as missing values, class imbalance, data heterogeneity, and more. D-ACE can be used to improve the dependability of machine learning algorithms by providing a detailed evaluation of the dataset and identifying any potential issues that may affect the performance of the algorithms. By addressing these issues and ensuring the quality of the dataset, the performance of machine learning algorithms can be improved, leading to more accurate and reliable results. In general, D-ACE can be a valuable tool for measuring the quality of datasets and ensuring the dependability of machine learning algorithms.</div>

Currently Supporting Characteristics:

* Dimensionality (d)
*	NrOfInstances (N)
*	NrOfClasses (C)
*	ZeroSparsity (OS)	
*	NaNSparsity (NS)
*	DataSparsity (DS)	
*	DataSparsityRatio (DSR)	
*	Correlation of Featues with Class (CorrFC)
*	Correlation of Featues without Class (CorrFNC)	
*	Multivariate normality? (MVN)	
*	Homogeneity of class covariance (HCCov)	
*	Intrinsic Dimensionality-PCA (ID)	
*	Intrinsic Dimensionality Ratio (IDR)	
*	Feature Noise variance (FN1)	
*	Feature Noise paper (FN2)	

To-do:
* adding dataset separability evaluation metrics
* adding geometric characteristics
