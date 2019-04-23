import numpy as np
from .. import activation_abs_c as activation_abs_c


class elu_act_c (activation_abs_c):
    
  def __init__ (self, alpha = 1):
    super ().__init__ ("elu");
    self.set_alpha (k);
    
  def set_alpha (self, alpha):
    self.set_key ("alpha", alpha);
  
  def get_alpha (self):
    return (self.get_key ("alpha"));

  def act_func (self, x, args = None):
    op = x.copy ();
    op[x<0] = (exp (op[x<0]) - 1) * self.get_alpha ();
    return op;

  def act_dash_func  (self, x, args = None):
    op = np.ones (x.shape);
    op[x<0] = self.act_func (op[x<0]) + self.get_alpha ();
    return op;
  
  
