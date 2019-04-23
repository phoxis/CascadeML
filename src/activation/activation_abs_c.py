import abc
import numpy as np

class activation_abs_c (metaclass = abc.ABCMeta):

  __args = None;
  __name = None;
  
  def __init__ (self, name = None):
    self.__args = dict ();
    self.__name = name;
    #if (type (self) is "activation_abs_c"):
      #raise Exception ("Cannot instantise. activation_abs_c is an abstract class.");
  
  def act (self, x, args = None):
    self.__validate (x, args);
    return (self.act_func (x, args));
  
  
  def act_dash (self, x, args = None):
    self.__validate (x, args);
    return (self.act_dash_func (x, args));
  
  
  def __validate (self, x, args):
    assert (type (x) is np.ndarray);
    assert (type (args) is dict or args is None);
  
  def get_key (self, key):
    if (key in self.__args):
      return self.__args[key];
    else:
      return None;
    
  def get_name (self):
    return self.__name;
    
  def set_key (self, key, value):
    self.__args[key] = value;
    
  # To override in child class with the implementation of the threshold function
  @abc.abstractmethod
  def act_func (self, x, args = None):
    pass;
  
  # To override in child class with derivative of the threshold function
  @abc.abstractmethod
  def act_dash_func (self, x, args = None):
    pass;

    
