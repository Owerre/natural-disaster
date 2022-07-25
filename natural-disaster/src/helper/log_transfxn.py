#############################
# Author: S. A. Owerre
# Date modified: 12/03/2021
# Class: Log Transformation
##############################

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
  """This class performs log(1+x) transformation on numerical features."""

  def __init__(self):
    """Define parameters."""

  def fit(self, X, y=None):
    """Do nothing."""
    return self

  def transform(self, X, y=None):
    """Log transform numerical variables."""
    num_attribs = list(X.select_dtypes('number'))
    self.X_num = X[num_attribs]
    return np.log1p(self.X_num )