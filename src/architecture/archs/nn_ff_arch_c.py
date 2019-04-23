# NOTE: For cost function, in the neural network, keep a way such that the error can be reported using other metrics than what is used to compute the backpropagation gradient
# NOTE: Think and keep a provision for adding and removing layer while training.
# NOTE: add_unit_in_layer, remove_unit_in_layer, Maybe not right now, but keep a provision in the backpropagation such that we can add and/or remove units while training.
# NOTE: Think and keep a provision for dropouts
# NOTE: When writing backpropagation, keep a provision such that one can add remove layer or units between iterations or batches
# NOTE: No assertion now. Decide on more assertion layer

import numpy as np
from copy import deepcopy

# TODO: Make gradient computation here. The architecture should know it's gradient. Should have an update gradient function.

class nn_ff_arch_c (nn_arch_abs_c):
    
  """
  function name: '__init__'
  args: 
          
  return:
          None
  """
  def __init__ (self, arch = None, nhid_units = None, activations = None, W_init_func = None):
    super ().__init__ ("feedforward multi-layer neural network");
    # TODO: Copy constructor?
    
    # Private variables
  
    ############################
  
    self.__nhid_units = list ();   # Array of integers indicating the number of units per layer. 
    self.__W = list ();            # Weights. A list os nympy 2d arrays.
    self.__W_init_func = list ();  # Function pointer which is used to initialise W. Should accept two arguments, rows and cols, and return an np.array;
    self.__op = list ();           # Outputs at each layer. A list of numpy 1d arrays.
    self.__op_unthresh = list ();  # Outputs at each layer BEFORE applying the threshold.
    self.__dropout_mask = list (); # Dropout mask for a given iteraation
    self.__activation_list = list ();          # Activation for each layer. A list of activation_c class objects.
    self.__bias = None;
    self.__dropout_prob = False;       # Probability of dropout
    # Weight update object, which knows how to do the weight update given the type of update.
    #NOTE: Not used right now. This can be useful to have disabled connections, or non-traininable connections fixed by the architecture.
    #TODO: Should be an abstract class, then we instantise, also with a factory.
    self.__W_update_obj = list (); 
    
    #################
    
    if (isinstance (arch, nn_ff_arch_c) is True):
      self.set_layers (arch.get_per_layer_units ().copy (), arch.get_per_layer_activation ().copy ());# FIXME: Need a copy routine.
      self.set_bias (arch.get_bias ().copy ());
      self.set_per_layer_weight (deepcopy (arch.get_per_layer_weight ()));
      #self.set_per_layer_weight ([w.copy () for w in arch.get_per_layer_weight ()]);
    else:  
      if (nhid_units == None and activations == None and W_init_func == None):
        pass;
      elif (nhid_units != None and activations != None):
        self.set_layers (nhid_units, activations, W_init_func);
        self.__bias = [1] * (len (nhid_units) - 1);
      else:
        pass;
    
  def copy_from (self, arch):
      self.set_layers (arch.get_per_layer_units ().copy (), arch.get_per_layer_activation ().copy ());# FIXME: Need a copy routine.
      self.set_bias (arch.get_bias ().copy ());
      self.set_per_layer_weight (deepcopy (arch.get_per_layer_weight ()));
    
  
  def copy (self):
    pass;
    
  # TODO: an Init weight should go here, which will be called by the learner
    
  
  # This takes the number of units, activation function, and the weight initialisation function and a layer index indicating where to add this layer. "None" defaults to append
  # Using this function we can write otherr helper functions like 'set_layers'. Therefore remove set_layer from below, and write this function, and call this function in set_layer
  # FIXME: Modify so that we can add a layer at any 'loc'. NOTE: We can also allow inserting a new input or output layer, this would be interesting.
  def add_layer (self, units, activation_obj, W_init_func = None, loc = None):
    assert (units > 0), "Argument 'units' should be a positive integer";
    assert (isinstance (activation_obj, activation.activation_abs_c)), "Argument 'activation_obj' should be an object of a subclass of 'activation.activation_abs_c'";
    assert ((len (self.__nhid_units) >= 2) and (len (self.__activation_list) >= 2)), "Define feedforward architecture input units and output units before adding hidden layers";
      
    self.__nhid_units.extend (units);
    self.__activation_list.append (activation_obj);
    
    # If atleast two layers added (input and output), set the weights
    if (len (self.__nhid_units) >= 2):
    
      if (W_init_func is None):
        self.__W_init_func = np.random.rand;
        
      assert (sum (list (map (lambda x: type (x) is np.ndarray, W_init_func))) != len (W_init_func)), "Argument 'W_init_func' should be a callable, or numpy ndarray";
      mrows, mcols = self.__nhid_units[len (self.__nhid_units) - 2] + 1, self.__nhid_units[len (self.__nhid_units) - 1];
      if (type (W_init_func) is np.ndarray):
        assert (W_init_func.shape == (mrows, mcols)), "Argument 'W_init_func' has incorrect dimensions";
        self.__W_init_func = W_init_func;
        
      elif (callable (self.__W_init_func) is True):
        self.__W.append (self.__W_init_func (mrows, mcols));
        
  # TODO: Trying to add units to existing layer. Need more general function
  def add_units (self, layer, units, W_init_func = None):
    
    r1, c1 = self.__W[layer-1].shape;
    r2, c2 = self.__W[layer].shape;
    self.__W[layer-1] = np.vstack (self.__W[layer-1][:,0:(r1-1)], self.__W_init_func (c1, units) ,self.__W[layer-1][:,r1-1]);
    self.__W[layer  ] = np.vstack (self.__W[layer  ][0:(r2-1),:], self.__W_init_func (units, c2) ,self.__W[layer  ][r2-1,:]);
    self.__nhid_units[layer] += units;
    
    
    
  # TODO: Remove a specific layer. Right now we need not allow removing input or outpupt layer. But it would be 
  # interesting in future to also allowing removing the input or output layer and replace the input or output
  # layer by the immediate next layer.
  def remove_layer (self, loc = None):
    pass;
  
  
  # TODO: This should give the layer description, explaining the architecture, for example "4 - sigmoid - 5 - linear - 6 sigmoid" or something similar
  def get_layer_description ():
    
    pass;
  
  # TODO: Add a unit in a layer.
  def add_unit_in_layer ():
    
    pass;
  
  # TODO: Remove a unit in a layer
  def remove_unit_in_layer ():
    
    pass;
  
  #@classmethod
  def set_layers (self, nhid_units, activations, W_init_func = None):
    
    assert type (nhid_units) in [list, np.ndarray], "Argument 'nhid_units' should be a list or np.ndarray of integers, each element indicating the number of units in the corresponding layer";
    assert sum ([type (b) is int for b in nhid_units]) == len (nhid_units), "Argument 'nhid_units' should be a list or np.ndarray of integers, each element indicating the number of units in the corresponding layer";
    assert (isinstance(activations, activation.activation_abs_c) or # Either activations is just a function
              ((len (activations) == (len (nhid_units) - 1)) and (sum ([isinstance(a, activation.activation_abs_c) for a in activations]) == len (activations)) )), "Argument 'activations' should either be an object of a subclass of 'activations.activation_abs_c', or a list of objects of subclass of 'activations.activation_abs_c', where the list has length than 'nhid_units'";
    # TODO: W is the set of trainable variable. Better to isolate from other variables. Then this can be handled differently than the other variables.
    #assert (type (W_init_func) is list) and (len (W_init_func) == (len (nhid_units) - 1)), "Argument 'W' should be None, or initial values for the weights of each layer";
      
    # If one instance then copy the same activation function to all layers. Else set set the list of activations.
    if (isinstance(activations, activation.activation_abs_c) == True):
      self.__activation_list = [activations] * (len (nhid_units) - 1);
    else:
      self.__activation_list = activations;
    
    self.__nhid_units = nhid_units.copy ();
    self.init_weights (W_init_func);
    self.__bias = [1] * len (nhid_units);
    
    
  # TODO: Decide what the self.__W_init_func should hold. Do we keep on holding?
  #@classmethod
  def init_weights (self, W_init_func = None):

    if (W_init_func is None):
      self.__W_init_func = np.random.rand;
    else:
      assert (callable (W_init_func) == True) or (sum (list (map (lambda x: type (x) is np.ndarray, W_init_func))) == len (W_init_func)), "Argument 'W_init_func' should be a callable, list of numpy ndarray";
      self.__W_init_func = W_init_func;
    
    if (type (self.__W_init_func) is list):
      self.__W = list ();
      for this_layer_idx in range (len (self.__nhid_units) - 1):
        mrows, mcols = self.__nhid_units[this_layer_idx] + 1, self.__nhid_units[this_layer_idx + 1];
        assert (self.__W_init_func[this_layer_idx].shape == (mrows, mcols)), "Argument 'W_init_func' has incorrect dimensions for layer connection between " + str (this_layer_idx) + " and " + str (this_layer_idx + 1) + ". " + str (mrows) + ", " + str (mcols) + " and " + str (self.__W_init_func[this_layer_idx].shape);
        self.__W.append (self.__W_init_func[this_layer_idx].copy ());
      
    elif (callable (self.__W_init_func) is True):
      self.__W = list (); # Length total layers - 1
      for this_layer_idx in range (len (self.__nhid_units) - 1):
        mrows, mcols = self.__nhid_units[this_layer_idx] + 1, self.__nhid_units[this_layer_idx + 1];
        self.__W.append (self.__W_init_func (mrows, mcols));
    else:
      print ("Not initialising weights\n");
      pass;
    
  
  """
  function name: 'do_forward_pass'
  args: 
          'X': An 'numpy.ndarray' type representing datapoints. Each row should represent a datapoint.
  return:
          None
  description:
          Forward propagates the dataset "X" through the network, and stores the output at each layer in '__op'
          These outputs can be fetched using 'get_per_layer_output'
  """  
  def do_forward_pass (self, X, train = True):
    
    assert (type (X) is np.ndarray), "Argument 'X' should be a type of np.ndarray representing the data for forward propagate";
    assert (X.shape[1] == self.__nhid_units[0]), "Argument 'X' have incorrect dimensions";
    
    layer_count           = len (self.__nhid_units);
    self.__op             = [None] * layer_count;
    self.__op_unthresh    = [None] * layer_count;
    self.__dropout_mask   = [None] * layer_count;
    self.__op[0]          = X;
    self.__op_unthresh[0] = X;
    
    if (self.__dropout_prob is not None) and (train is True):
        self.__dropout_mask[0] = np.random.binomial (1, self.__dropout_prob, size = self.__op_unthresh[0].shape[1]) / self.__dropout_prob;
        self.__op_unthresh[0] *= self.__dropout_mask[0];
        
    for this_layer_idx in range (layer_count - 1):
      self.__op_unthresh[this_layer_idx + 1] =  np.matmul ( np.insert (self.__op[this_layer_idx], self.__op[this_layer_idx].shape[1], np.array ([self.__bias[this_layer_idx]]), axis = 1), self.__W[this_layer_idx] );
      self.__op[this_layer_idx + 1] = self.__activation_list[this_layer_idx].act (self.__op_unthresh[this_layer_idx + 1]);
      
      if (self.__dropout_prob is not None) and (train is True):
        self.__dropout_mask[this_layer_idx + 1] = np.random.binomial (1, self.__dropout_prob, size = self.__op_unthresh[this_layer_idx + 1].shape[1]) / self.__dropout_prob;
        self.__op_unthresh[this_layer_idx + 1] *= self.__dropout_mask[this_layer_idx + 1];
        
      
    # FIXME: Confirm the index operations do what exactly we want. Also make sure if this makes any sense or not.
    if (self.get_fixed_output_unthresh () is not None):
      #self.__op_unthresh[layer_count - 1] = (self.__op_unthresh[layer_count - 1] + self.get_fixed_output_unthresh ())/2;
      self.__op_unthresh[layer_count - 1] += self.get_fixed_output_unthresh ();
      self.__op[layer_count - 1] = self.__activation_list[layer_count - 2].act (self.__op_unthresh[layer_count - 1]);
      
  """
  function name: 'predict'
  args: 
          'X': An 'np.ndarray' type representing datapoints. Each row should represent a datapoint.
  return:
          An 'numpy.ndarray' type representing the output at the last layer of the feedforward architecture
  description: 
          Implements the abstract function in 'nn_arch_abs_c'. Calls 'do_forward_pass' to perform the forward
          pass and returns the values at the output layer.
  """  
  def predict (self, X):
    
    self.do_forward_pass (X, train = False);
    return self.__op[len (self.__nhid_units) - 1]; # Return reference. Faster.
  
  
  
  # NOTE: Below funcitons, does the work which we want to do, but is it a good design?
  # These functions expose the required information needed for the training algorithm
  # Is multiple inheritence an answer? OR something like a mediator pattern?
  
  """
  function name: 'get_per_layer_output'
  args:
        None
  return:
        A reference to 'list' type, with each element of the list being a 'numpy.ndarray' type
        representing the output at each layer
  description:
        Returns the outputs at each layer. Called by the training algorithms to access outputs
  """
  def get_per_layer_output (self):
    return self.__op;
  
  
  def get_per_layer_unthresh_output (self):
    return self.__op_unthresh;
  
  """
  function name: 'get_per_layer_weight'
  args:
        None
  return:
        A reference to 'list' type, with each element of the list being a 'numpy.ndarray' type
        representing the weight matrix at each layer
  description:
        Returns the weight matrix of each layer. Called by the training algorithms to access weights
  """
  def get_per_layer_weight (self):
    return self.__W; # Return reference.
  
  """
  function name: 'get_per_layer_activation'
  args:
        None
  return:
        A reference to 'list' type, with each element of the list being a 'activation.activation_abs_c' type
        representing the activation function at each layer
  description:
        Returns the activation function objects at each layer. Called by the training algorithms to 
        access the activation funcitons to find derivatives
  """
  def get_per_layer_activation (self):
    return self.__activation_list;
  
  
  """
  function name: 'get_per_layer_units'
  args:
        None
  return:
        A list of integers describing the units per layer
  description:
        The returned list per-layer unit count does not include the bias unit.
  """
  def get_per_layer_units (self):
    return self.__nhid_units; # Return reference.
  
  
  """
  function name: 'get_layer_count'
  args:
        None
  return:
        An integer type representing the number of layers
  description:
        Does not count the input units as a layer. If there are 1 hidden layer,
        this will return 2, one hidden and one output layer.
  """
  def get_layer_count  (self):
    return (len (self.__nhid_units) - 1);
  
  
  """
  function name: 'get_units_at_layer'
  args:
        'layer_no': integer type, indicating number
  return:
        An integer type representing the number of units at a given layer number 'layer_no'
  description:
        Get the number of units at the given layer. '0' indicates the input layer units.
        Does not include the bias ubit.
  """
  def get_units_at_layer (self, layer_no):
    return self.__nhid_units[layer_no];
  

  """
  function name: 'set_w'
  args:
        W: A 'list' of 'numpy.ndarray', where each list location indicates the weight matrix
           of the corresponding layer. This should be exactly in the format as in 'self.__W'
  return:
        None
  description:
        Sets the weights for the architecture. Called by training algorithms.
  """
  # TODO: Need validation for dimension checking
  def set_per_layer_weight (self, W):
    self.__W = W;
    
  def get_bias (self):
    return self.__bias;
  
  def set_bias (self, bias):
    assert (len (bias) == len (self.__nhid_units)), "Argument \"bias\" should have the same length as number of layers";
    self.__bias = bias;
  
  def get_architecture (self):
    return self.__nhid_units.copy ();
   
  # Implementing abstract function
  def get_layer_types (self):
    return ["fullconn"] * len (self.__W);
  
  # Implementing abstract function
  # Does not do anything now, as we do not implement automatic growth yet
  def set_this_layer_type (self, node_type):
    pass;
  
  
  def get_dropout_mask (self):
    return self.__dropout_mask;
  
  def set_dropout_prob (self, p):
    self.__dropout_prob = p;
    # Not implementing the getter
  
  # Overriding
  def get_weights (self):
    return np.concatenate (list (map (lambda x: x.flatten (), self.__W)));
