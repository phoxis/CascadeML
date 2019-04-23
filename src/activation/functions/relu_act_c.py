import numpy as np
from .. import activation_abs_c as activation_abs_c


class relu_act_c (activation_abs_c):
    
  def __init__ (self, k = 1):
    super ().__init__ ("relu");
    
  def act_func (self, x, args = None):
    op = x.copy ();
    op[x<=0] = 0;
    return op;

  def act_dash_func  (self, x, args = None):
    op = np.ones (x.shape);
    op[x<=0] = 0;
    return op;
  
