import numpy as np
from .. import activation_abs_c as activation_abs_c

# FIXME: We really do not need to check the args everyt time we call act_func, it makes unnecessary checks.

class sigmoid_act_c (activation_abs_c):
    
  def __init__ (self, k = 1):
    super ().__init__ ("sigmoid");
    self.set_k (k);
  
  def set_k (self, k):
    self.set_key ("k", k);
  
  def get_k (self):
    return (self.get_key ("k"));
  
  # If the caller passes the argument 'k', then use it. Else use the 
  # object's set method
  def act_func (self, x, args = None):
    k_val = self.get_k ();
    if (k_val is None):
      k_val = args["k"];
    return (1 / (1 + np.exp (-x * k_val)));

  # If the caller passes the argument 'k', then use it. Else use the 
  # object's set method
  def act_dash_func  (self, x, args = None):
    return self.act_func (x, args) * (1 - self.act_func (x, args));
