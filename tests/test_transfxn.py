#####################################
# Author: S. A. Owerre
# Date modified: 21/07/2022
# Function: Test for Transformation
######################################

import pandas as pd
import numpy as np
import pytest
import sys
base_path = ''
sys.path.append(base_path+'imbalanced-learning/imbalanced-learning/src/helper/')
import transfxn as tfxn

@pytest.fixture()
def input_df():
    X_train = pd.DataFrame(
        {
            'v_0':np.random.uniform(low=10, high=100, size=10,),
            'v_1':np.random.uniform(low=10, high=100, size=10,),
            'v_2':np.random.uniform(low=10, high=100, size=10,),
            'v_3':np.random.uniform(low=10, high=100, size=10,),
            'v_4':np.random.uniform(low=10, high=100, size=10,),
            'v_5':np.random.uniform(low=10, high=100, size=10,),
            'v_6':np.random.uniform(low=10, high=100, size=10,),
            'v_7':['r', 'g','b','r', 'g','b','r', 'g','b','b',],


        }
    )

    X_test = pd.DataFrame(
        {
            'v_0':np.random.uniform(low=10, high=100, size=7,),
            'v_1':np.random.uniform(low=10, high=100, size=7,),
            'v_2':np.random.uniform(low=10, high=100, size=7,),
            'v_3':np.random.uniform(low=10, high=100, size=7,),
            'v_4':np.random.uniform(low=10, high=100, size=7,),
            'v_5':np.random.uniform(low=10, high=100, size=7,),
            'v_6':np.random.uniform(low=10, high=100, size=7,),
            'v_7':['r', 'g','b','r', 'g','b','r',],


        }
    )
    return X_train, X_test

def test_transfxn(input_df):
    """Test preprocessing with both 
    numerical and categorical variables.
    """
    model = tfxn.TransformationPipeline()
    X_train, X_test = input_df
    X_train_scaled, X_test_scaled, feat_nm = model.preprocessing(X_train, X_test)
    num_cols_expected_train = X_train.shape[1] + len(X_train.v_7.unique())-1
    num_cols_expected_test = X_test.shape[1] + len(X_test.v_7.unique())-1
    assert X_train_scaled.shape == (X_train.shape[0], num_cols_expected_train)
    assert X_test_scaled.shape == (X_test.shape[0], num_cols_expected_test)
    assert len(feat_nm) == X_train.shape[1] + len(X_train.v_7.unique())-1

