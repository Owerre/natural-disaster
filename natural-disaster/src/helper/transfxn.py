#################################
# Author: S. A. Owerre
# Date modified: 12/03/2021
# Class: Transformation Pipeline
#################################

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TransformationPipeline:
    """This class is used for data preprocessing and transformations."""

    def __init__(self):
        """Define parameters."""

    def num_pipeline(self, X_train, X_test):
        """Transformation pipeline of data with only numerical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformation pipeline and transformed data in array
        """
        # Create pipeline
        num_pipeline = Pipeline(
            [
                ('p_transf',PowerTransformer(standardize = False)),
                ('std_scaler', StandardScaler()),
            ]
        )

        # Original numerical feature names 
        feat_nm = list(X_train.select_dtypes('number'))

        # Fit transform the training set and transform the test set
        X_train_scaled = num_pipeline.fit_transform(X_train)
        X_test_scaled = num_pipeline.transform(X_test)
        return X_train_scaled, X_test_scaled, feat_nm
    
    def cat_encoder(self, X_train, X_test):
        """Transformation pipeline of categorical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        ------
        Transformation pipeline and transformed data in array
        """
        # instatiate class
        one_hot_encoder = OneHotEncoder(handle_unknown = 'ignore')

        # fit transform the training set and transform the test set
        X_train_scaled = one_hot_encoder.fit_transform(X_train)
        X_test_scaled = one_hot_encoder.transform(X_test)

        # feature names for output features
        feat_nm = list(one_hot_encoder.get_feature_names_out(list(
            X_train.select_dtypes('O'))
            )
        )
        return X_train_scaled.toarray(), X_test_scaled.toarray(), feat_nm
  
    def preprocessing(self, X_train, X_test):
        """Transformation pipeline of data with 
        both numerical and categorical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformed data in array
        """

        # numerical transformation pipepline
        num_train, num_test, num_col = self.num_pipeline(
            X_train.select_dtypes('number'), 
            X_test.select_dtypes('number')
        )

        # categorical transformation pipepline
        cat_train, cat_test, cat_col = self.cat_encoder(
            X_train.select_dtypes('O'), 
            X_test.select_dtypes('O')
        )

        # transformed training and test sets
        X_train_scaled = np.concatenate((num_train,cat_train), axis = 1)
        X_test_scaled = np.concatenate((num_test,cat_test), axis = 1)

        # feature names
        feat_nm = num_col + cat_col
        return X_train_scaled, X_test_scaled, feat_nm
