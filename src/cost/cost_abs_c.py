import abc
import numpy as np

class cost_abs_c (metaclass = abc.ABCMeta):

  args = None;
  
  def __init__ (self):
    self.args = dict ();
    #if (type (self) is "cost_abs_c"):
      #raise Exception ("Cannot instantise. cost_abs_c is an abstract class.");
  
  def cost (self, output, target, args = None):
    self.__validate (output, target, args);
    return (self.cost_func (output, target, args));
  
  
  def cost_dash (self, output, target, args = None):
    self.__validate (output, target, args);
    return (self.cost_dash_func (output, target, args));
  
  
  def __validate (self, output, target, args):
    assert (type (output) is np.ndarray);
    assert (type (output) is np.ndarray);
    assert (type (args) is dict or args is None);
  
  def get_key (self, key):
    if (key in self.args):
      return self.args[key];
    else:
      return None;
    
  def set_key (self, key, value):
    self.args[key] = value;
    
  # To override in child class with the implementation of the threshold function
  @abc.abstractmethod
  def cost_func (self, output, target, args = None):
    pass;
  
  # To override in child class with derivative of the threshold function
  @abc.abstractmethod
  def cost_dash_func (self, output, target, args = None):
    pass;
  
  # Right now, optional to override. This is to return row-wise errors, one-hot encoded column-wise errors
  def cost_func_by_axis (self, output, target, axis, args = None):
    assert False, "'cost_func_by_axis' is not implemented";
    pass;
  
