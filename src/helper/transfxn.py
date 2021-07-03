####################################
# Author: S. A. Owerre
# Date modified: 10/05/2021
# Class: Transformations
####################################

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import numpy as np

# Helps with importing functions from different directory
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

# import custom class
from helper import log_transfxn as cf 

# Data pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

class TransformationPipeline:
    """
    A class for transformation pipeline
    """

    def __init__(self):
        """
        Define parameters
        """

    def num_pipeline(self, X_train, X_test):
        """
        Transformation pipeline of data with only numerical variables

        Parameters
        ___________
        X_train: Training feature matrix
        X_test: Test feature matrix

        Returns
        __________
        Transformation pipeline and transformed data in array
        """
        # Create pipeline
        num_pipeline = Pipeline([('p_transform', PowerTransformer(standardize=False)),
                                ('std_scaler', StandardScaler()),
                                ])

        # Original numerical feature names 
        feat_nm = list(X_train.select_dtypes('number'))

        # Fit transform the training set
        X_train_scaled = num_pipeline.fit_transform(X_train)
        
        # Only transform the test set
        X_test_scaled = num_pipeline.transform(X_test)
        return X_train_scaled, X_test_scaled, feat_nm
    
    def cat_encoder(self, X_train, X_test):
        """
        Transformation pipeline of categorical variables

        Parameters
        ___________
        X_train: Training feature matrix
        X_test: Test feature matrix

        Returns
        __________
        Transformation pipeline and transformed data in array
        """
        # Instatiate class
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        # Fit transform the training set
        X_train_scaled = one_hot_encoder.fit_transform(X_train)
        
        # Only transform the test set
        X_test_scaled = one_hot_encoder.transform(X_test)

        # Feature names for output features
        feat_nm = list(one_hot_encoder.get_feature_names(list(X_train.select_dtypes('O'))))
        return X_train_scaled.toarray(), X_test_scaled.toarray(), feat_nm
  
    def preprocessing(self, X_train, X_test):
        """
        Transformation pipeline of data with both numerical and categorical 
        variables.

        Parameters
        ___________
        X_train: Training feature matrix
        X_test: Test feature matrix

        Returns
        __________
        Transformed data in array
        """

        # Numerical transformation pipepline
        num_train, num_test, num_col = self.num_pipeline(X_train.select_dtypes('number'), 
                                        X_test.select_dtypes('number'))

        # Categorical transformation pipepline
        cat_train, cat_test, cat_col = self.cat_encoder(X_train.select_dtypes('O'), 
                                        X_test.select_dtypes('O'))

        # Transformed training set
        X_train_scaled = np.concatenate((num_train,cat_train), axis = 1)

        # Transformed test set
        X_test_scaled = np.concatenate((num_test,cat_test), axis = 1)

        # Feature names
        feat_nm = num_col + cat_col
        return X_train_scaled, X_test_scaled, feat_nm