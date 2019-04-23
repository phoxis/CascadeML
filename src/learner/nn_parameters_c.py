# Holds and manages the parameters for neural networks.
import dill as pickle

class nn_parameters_c:
  
  def __init__ (self, params = None):
    
    self.__param_hash = dict ();
    
    if (params != None):
      self.update_params (params);
    else:
      self.__param_hash = dict ();
  
  def get_param (self, arg_key):
    if (arg_key not in self.__param_hash):
      return None;
    else:
      return self.__param_hash[arg_key];

  # Set one parameter
  def set_param (self, arg_key, arg_val):
    self.__param_hash[arg_key] = arg_val;
  
  # Update a set of parameters
  def update_params (self, new_params):
    if (type (new_params) is dict):
      self.__param_hash = {**self.__param_hash, **new_params};
    elif (isinstance (new_params, nn_parameters_c)):
      self.__param_hash = new_params.__param_hash.copy ();

  # Clear the dict
  def reset_params (self):
    self.__param_hash = dict ();
 
  # Return a string version of the parameters
  def get_param_to_string (self):
    # The pickle/dill dump and load makes the function references as strings.
    # TODO: Possibly later also store the function's code or name or description.
    # To do this, we can enforce that instead of a function one must pass a class which has two components
    # one is the function and another one is the cost. Also, maybe remove dependency on yaml and pickle here?
    # Need indentation as well
    return pickle.loads (pickle.dumps (self.__param_hash, 0));
  
