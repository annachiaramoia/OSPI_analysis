# Online Shopper Purchase Intention

## Overview

This project implements an end-to-end machine learning pipeline for analyzing online shopping sessions and predicting whether a session results in a purchase.

The repository includes the original dataset, the derived training and test sets, and the complete analysis and modeling workflow in `Pipeline.ipynb`.

## Workflow

The project is organized into three main stages:

### 1. Data Exploration and `ExitRates` Imputation

The pipeline begins with exploratory data analysis to inspect data quality, feature distributions, missing values, correlations, and the distribution of the `Revenue` target.

Since `ExitRates` contains missing observations, its recovery is treated as a regression problem. Multiple regression approaches are evaluated, including Linear Regression, forward feature selection, Polynomial Regression, LASSO, and Ridge. The resulting regression pipeline is then used to impute missing `ExitRates` values.

### 2. Revenue Classification

The completed dataset is used to model the binary `Revenue` target. The classification stage considers Logistic Regression, regularized Logistic Regression, Discriminant Analysis, K-Nearest Neighbors, and Support Vector Machines.

Preprocessing, feature scaling, feature selection, cross-validation, and hyperparameter tuning are integrated into scikit-learn pipelines to ensure a consistent modeling process and reduce data leakage.

The classification workflow is also repeated without `ExitRates` to evaluate the contribution of the imputed feature.

### 3. Clustering Analysis

The final stage explores the data from an unsupervised perspective using Agglomerative Clustering, K-Means, and DBSCAN. The resulting clusters are compared with `Revenue` to assess whether natural session groupings align with purchasing behavior.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, and scikit-learn.
