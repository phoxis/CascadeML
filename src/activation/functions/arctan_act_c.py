import numpy as np
from .. import activation_abs_c as activation_abs_c


class arctan_act_c (activation_abs_c):
    
  def __init__ (self, k = 1):
    super ().__init__ ("arctan");
    self.set_k (k);

  # NOTE: Not using any argument right now
  def set_k (self, k):
    self.set_key ("k", k);
  
  def get_k (self):
    return (self.get_key ("k"));
  
  def act_func (self, x, args = None):
    return np.arctan (x);

  def act_dash_func  (self, x, args = None):
    return 1 / (x**2 + 1);
  
  
  
