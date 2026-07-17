# Introduction

## What is D-ACE?

[![License: MIT](https://img.shields.io/github/license/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics?color=yellow)](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics/blob/main/pyproject.toml)
[![Package](https://img.shields.io/badge/package-dace-brightgreen)](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics)
[![Research](https://img.shields.io/badge/focus-dataset%20quality-purple)](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics)

D-ACE, Dataset Assessment and Characteristics Evaluation, helps researchers and
machine learning engineers inspect the quality and structure of tabular datasets
before training models. It reports compact and repeatable dataset
characteristics such as dimensionality, sparsity, class count, feature
correlation, multivariate normality, intrinsic dimensionality, feature noise,
and homogeneity of class covariance.

Use D-ACE as an early dataset assessment step when you need to understand
whether data shape, missingness, class structure, or feature relationships may
affect model dependability.

!!! tip

    Run D-ACE before model selection. A short characteristic report can reveal
    high sparsity, unexpected class structure, or correlated features before
    those issues become harder to debug in model behavior.

## How do I install D-ACE?

Install the package from the repository root:

```bash
git clone https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics.git
cd Dataset-Characteristics
python -m pip install .
```

For local development, install the editable package with development tools:

```bash
python -m pip install -r requirements-dev.txt
```

## Quick usage

The all-in-one report function returns a `pandas.DataFrame` with each supported
characteristic and its calculated value.

```python
import pandas as pd

from dace.characteristics.dataset import getAllDatasetCharacteristicsTable

dataset = pd.read_csv("data/iris.csv")
report = getAllDatasetCharacteristicsTable(dataset, className="Species")

print(report.to_string(index=False))
```

For lower-level workflows, import individual functions from
`dace.characteristics.dataset` and combine them with utilities from
`dace.utils.data`.

## What can D-ACE assess?

<div class="grid cards" markdown>

-   **Dataset structure**

    ---

    Measure the number of features, instances, and classes to understand the
    basic shape of a dataset.

    `getDatasetDimensionality`, `getDatasetNumberOfInstances`,
    `getDatasetNumberOfClasses`

-   **Sparsity and missingness**

    ---

    Inspect zero sparsity and NaN sparsity so preprocessing decisions are based
    on observable data quality signals.

    `getZeroSparsity`, `getNaNSparsity`

-   **Feature relationships**

    ---

    Summarize feature correlation with and without class labels, then check
    multivariate normality for the feature space.

    `getCorrelationOfFeaturesWithClass`, `getMultiVariateNormality`

-   **Dimensionality and noise**

    ---

    Estimate intrinsic dimensionality with PCA-based methods and compare feature
    noise signals across datasets.

    `getIntrinsicDimensionaltiy`, `getFeatureNoise`

</div>

## Supported characteristics

| Characteristic | Short name | What it helps reveal |
| --- | --- | --- |
| Dimensionality | `d` | Number of feature columns, excluding the class column. |
| Number of instances | `N` | Dataset size available for training or analysis. |
| Number of classes | `C` | Label diversity in the selected class column. |
| Zero sparsity | `OS` | Fraction of zero values across feature entries. |
| NaN sparsity | `NS` | Fraction of missing values across feature entries. |
| Data sparsity | `DS` | Dataset sparsity estimate based on class count and dimensionality. |
| Data sparsity ratio | `DSR` | Placeholder in the current implementation; returned as `TBD`. |
| Correlation with class | `CorrFC` | Mean absolute feature correlation calculated within class groups. |
| Correlation without class | `CorrFNC` | Mean absolute feature correlation across features only. |
| Multivariate normality | `MVN` | Henze-Zirkler normality signal for the feature space. |
| Homogeneity of class covariance | `HCCov` | Similarity of covariance structure between classes. |
| Intrinsic dimensionality | `ID` | PCA-based intrinsic dimensionality estimate. |
| Intrinsic dimensionality ratio | `IDR` | Ratio between intrinsic dimensionality and observed dimensionality. |
| Feature noise variance | `FN1` | Mean variance across feature columns. |
| Feature noise paper metric | `FN2` | Noise estimate based on observed and intrinsic dimensionality. |

## Package map

| Module | Purpose | Common entry points |
| --- | --- | --- |
| `dace.characteristics.dataset` | Dataset quality and characteristic metrics. | `getAllDatasetCharacteristicsTable`, `getZeroSparsity`, `getFeatureNoise` |
| `dace.utils.data` | Data loading, column cleanup, scaling, PCA, splitting, and Random Forest helpers. | `readDataset2DataFrame`, `fixDatasetColumnName`, `scaleDatasetUsingStandardScalar` |
| `dace.display.data` | Display helpers for model diagnostics and dataset exploration. | `displayConfusionMatrixHeatMap`, `displayClassificationReport`, `displayCorrelationMatrixHeatMap` |

## Recommended workflow

1. Load a CSV dataset into a `pandas.DataFrame`.
2. Clean column names with `fixDatasetColumnName` when names contain spaces or
   mixed casing.
3. Choose the class or label column for the dataset.
4. Run `getAllDatasetCharacteristicsTable` for a complete characteristic report.
5. Review sparsity, correlation, dimensionality, and normality signals before
   selecting preprocessing steps or model families.

!!! note

    Most characteristic functions expect tabular data in a `pandas.DataFrame`.
    Functions that compare features with labels require the class column name
    via the `className` argument.

## Development

Run the package tests from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Build the documentation locally with MkDocs:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

## Roadmap

- Add dataset separability metrics.
- Add geometric dataset characteristics.
- Add a mislabeling ratio for label quality assessment.
- Explore data valuation methods such as Data Shapley.
- Add fairness-oriented balance checks for sensitive features.

## Collaborators

- [Dependable Intelligent Systems Lab, University of Hull](https://www.hull.ac.uk/work-with-us/research/groups/dependable-intelligent-systems)
- [Fraunhofer Institute for Experimental Software Engineering](https://www.iese.fraunhofer.de)

## Contributors

Jerin Antony, Akinwande Adegbola, Zhibao Mian, Septavera Sharvia,
Koorosh Aslansefat, Mohammad Naveed Akram, Iannis Sorokos, and
Yiannis Papadopoulos.

## FAQ

### What problem does D-ACE solve?

D-ACE creates a compact profile of a tabular dataset before model training. The
profile helps identify data quality and structure issues that may influence
machine learning performance and dependability.

### Does D-ACE train models?

D-ACE is primarily a dataset assessment package. The `dace.utils.data` module
includes helper functions for splitting data and training a Random Forest model,
but the core package focus is dataset characterization.

### Which data format should I use?

Use CSV files or any source that can be loaded into a `pandas.DataFrame`.
Several example CSV datasets are included in the repository under `data/`.

### Where can I contribute?

Read the repository guides for
[contributing](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics/blob/main/CONTRIBUTING.md),
[security](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics/blob/main/SECURITY.md),
and the
[code of conduct](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics/blob/main/CODE_OF_CONDUCT.md)
before opening issues or pull requests.

## License

D-ACE is available under the
[MIT License](https://github.com/Dependable-Intelligent-Systems-Lab/Dataset-Characteristics/blob/main/LICENSE).
