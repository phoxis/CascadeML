import numpy as np
from .. import cost_abs_c as cost_abs_c

class mse_cost_c (cost_abs_c):
  
  def __init__ (self):
    super ().__init__ ();
  
  #@classmethod
  def cost_func (self, output, target, args = None):
    output_errors = np.average((output - target) ** 2, axis=0);
    
    cost = None;
    if self.get_key ("wts") is not None:
      cost = np.sum (self.get_key ("wts") * output_errors) / len (output_errors);
    else:
      cost = np.average(output_errors);
      
    return cost;
    #return (np.mean (np.mean (np.power (output - target, 2), axis = 1)));

  #@classmethod
  def cost_dash_func (self, output, target, args = None):
    tmp1 = None;
    if self.get_key ("wts") is None:
      tmp1 = (output - target);
    else:
      tmp1 = self.get_key ("wts") * np.array (output - target);
    return tmp1;
  
  def cost_func_by_axis (self, output, target, axis, args = None):
    return np.mean (np.power (output - target, 2), axis = axis);
  
