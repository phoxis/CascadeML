import numpy as np
from .. import activation_abs_c as activation_abs_c

class linear_act_c (activation_abs_c):
  
  def __init__ (self):
    super ().__init__ ("linear");
  
  @classmethod
  def act_func (self, x, args = None):
    return x;

  @classmethod
  def act_dash_func (self, x, args = None):
    return np.ones (x.shape);
  
