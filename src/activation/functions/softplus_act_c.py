import numpy as np
from .. import activation_abs_c as activation_abs_c


class softplus_act_c (activation_abs_c):
    
  def __init__ (self):
    super ().__init__ ("softplus");
    
  def act_func (self, x, args = None):
    return np.log (1 + np.exp (x));

  def act_dash_func  (self, x, args = None):
    return (1 / (1 + np.exp (-1 * x)));
  
  
  
  
