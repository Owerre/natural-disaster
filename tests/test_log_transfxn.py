#######################################
# Author: S. A. Owerre
# Date modified: 21/07/2022
# Function: Test for Log Transformation
########################################

import pandas as pd
import numpy as np
import pytest
import sys
base_path = ''
sys.path.append(base_path + 'imbalanced-learning/imbalanced-learning/src/helper/')
import log_transfxn as lg

@pytest.fixture()
def input_df():
    return pd.DataFrame(
        {
            'v_0':np.random.uniform(low=10, high=100, size=10,),
            'v_1':np.random.uniform(low=10, high=100, size=10,),
            'v_2':np.random.uniform(low=10, high=100, size=10,),
            'v_3':np.random.uniform(low=10, high=100, size=10,),
            'v_4':np.random.uniform(low=10, high=100, size=10,),
            'v_5':np.random.uniform(low=10, high=100, size=10,),
            'v_6':np.random.uniform(low=10, high=100, size=10,),


        }
    )

def test_logtranfxn(input_df):
    model = lg.LogTransformer()
    X_log = model.fit_transform(input_df)
    assert X_log.shape == input_df.shape