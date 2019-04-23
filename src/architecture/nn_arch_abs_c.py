import abc

"""
Abstract class indicating the mandatory methods to be implemented.
The definition is very abstract just providing the base class for all
the different neural network architectures. To make a new architecture, 
for example, convolutional network, feed forward network, recurrent 
network etc. extend this class and implement 'predict'. Call constructor 
to set architecture name.
"""

class nn_arch_abs_c (metaclass = abc.ABCMeta):
  
  # Private
  __name = None;    # Name of the architecture
  
  
  ################


  """
  function name: '__init__'
  args: 
          'arch_name': str or NoneType. The name of the architecture.
  return:
          None
  """  
  def __init__ (self, arch_name = None):
    self.set_name (arch_name);
    
    self.__fixed_output_unthresh_train = None;
    self.__fixed_output_unthresh_valid = None;
    self.__fixed_output_unthresh = None; # Present one in use
  
  """
  function name: 'get_name'
  args: 
          none
  return:
          Returns the name of the architecture
  """
  def get_name (self):
    return self.__name;
  
  
  """
  function name: set_name
  args: 
          'arch_name': str or NoneType. The name of the architecture
  return:
          none
  """
  def set_name (self, arch_name):
    assert (type (arch_name) in (str, type (None))), "Argument 'arch_name' should be of type str or None";
    self.__name = arch_name;

  
  
  """
  function name: 'predict'. abstract.
  args: 
          'X': Input dataset of type 'numpy.ndarray'
  return:
          A 'numpy.ndarray' with the output predictionss
  """
  @abc.abstractmethod
  def predict (self, X):
    pass;
  
  @abc.abstractmethod
  def init_weights (self, winit_func):
    pass;
  
  
  # TODO: Do we need make some of the below abstract?
  def get_fixed_train_valid_op (self, set_type = "train"):
    if (set_type == "train"):
      return self.__fixed_output_unthresh_train;
    elif (set_type == "valid"):
      return self.__fixed_output_unthresh_valid;
  
  
  def set_fixed_train_valid_op (self, op_unthresh, set_type = "train"):
    if (set_type == "train"):
      self.__fixed_output_unthresh_train = op_unthresh;
    elif (set_type == "valid"):
      self.__fixed_output_unthresh_valid = op_unthresh;
      
  # FIXME: Make sure if this makes any sense or not.
  def set_fixed_output_unthresh (self, op_unthresh):
    self.__fixed_output_unthresh = op_unthresh;
    
  def get_fixed_output_unthresh (self):
    return self.__fixed_output_unthresh;
  
  
  def get_weights (self):
    return "\"get_weights\" not implemented";
  
  def get_architecture (self):
    return "\"get_architecture\" not implemented";
  
  
  def get_layer_types (self):
    return "\"get_layer_types\" not implemented";
  
  def set_this_layer_type (self, node_type):
    return "\"set_this_layer_type\" not implemented";
