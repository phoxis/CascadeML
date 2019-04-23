import numpy as np
from .. import activation_abs_c as activation_abs_c


class tanh_act_c (activation_abs_c):
    
  def __init__ (self, k = 1):
    super ().__init__ ("tanh");
    self.set_k (k);

  # NOTE: Not using any argument right now
  def set_k (self, k):
    self.set_key ("k", k);
  
  def get_k (self):
    return (self.get_key ("k"));
  
  def act_func (self, x, args = None):
    #return (2 / (1 + np.exp (-2 * x))) - 1;
    return np.tanh (x);

  def act_dash_func  (self, x, args = None):
    return 1 - np.power (self.act_func (x, args), 2);
  
