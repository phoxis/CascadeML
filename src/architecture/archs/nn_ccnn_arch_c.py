import numpy as np

# NOTE: The output layer can be more than one layer essentially. Therefore in that case __cW_o is also a list, essentially an nn_ff_arch type. Not to implement now.
# FIXME: Note that the state restore method in "nn_arch_abs_c" will not work with this class right now. Need the copy function to be implemented.

# Cascade correlation neural network architecture: https://papers.nips.cc/paper/207-the-cascade-correlation-learning-architecture.pdf
class nn_ccnn_arch_c (nn_arch_abs_c):
  

  def __init__ (self, arch = None, W_init = "urand", act_init = 'sigmoid'):
    super ().__init__ ("cascacde correlation architecture neural network");
    
    if (arch is not None):
      self.__op_f = deepcopy (arch.get_op_final ());
      self.__op_f_unthresh = deepcopy (arch.get_op_unthresh_final ());
      self.__op_h = deepcopy (arch.get_op_hidden_layer ());
      self.__op_h_unthresh = deepcopy (arch.get_op_unthresh_hidden_layer ()); 
      self.__cW_h = deepcopy (arch.get_per_layer_cascade_weight ());
      self.__cW_o = deepcopy (arch.get_op_layer_weight ()); # A matrix. Once, __op_h is fully computed, this will hold the final matrix which does the final layer pass. This is grown by the training algorithm.
      self.__activation_h_list = arch.get_per_layer_cascade_activation (); #FIXME: Need a copy # List of activations at each cascaded layer, objects of 'activation_c' class. This can be populated by the learning algorithm.
      self.__activation_h_preset = arch.get_activation_h_preset (); #FIXME: Need a copy  # This is the type of activation function, an object of 'activation_c' which is preset for this architecture. If all the cascaded layers have the same function, then this is copied to '__activation_h_list'. 
      self.__activation_op = arch.get_output_activation () #FIXME: Need a copy;        # Activation of the output layer.
      self.__max_layers = deepcopy (arch.get_max_layers ()); # Holds the number of cascaded layers in the architecture. Determined by the learning algorithm. Not used. Controlled by the maxepoch of outer loop.
      self.__c_nhid_units = deepcopy (arch.get_cascade_nhid_units ());      # TODO: List holding the number of hidden unit per cascaded layer. To be populated by the learning algorithm.
      self.__ip_op = deepcopy (arch.__get_ip_op ());  # HACK: Quick storage of input and output
      self.__cascade_node_types = deepcopy (arch.get_layer_types ());
    else:
      self.__op_f = None; # A matrix for the final output.
      self.__op_f_unthresh = list (); # A list of np.ndarray, each having the output of the corresponding layer.
      self.__op_h = list (); # A list of np.ndarray, each having the output of the corresponding layer.
      self.__op_h_unthresh = None; 
      self.__cW_h = list (); # A list of hidden layer matrices, where __cW_h[i] indicates the ith cascaded layer's input-to-hidden layer weight matrix.
      self.__cW_o = None; # A matrix. Once, __op_h is fully computed, this will hold the final matrix which does the final layer pass. This is grown by the training algorithm.
      self.__activation_h_list = list (); # List of activations at each cascaded layer, objects of 'activation_c' class. This can be populated by the learning algorithm.
      self.__activation_h_preset = None;  # This is the type of activation function, an object of 'activation_c' which is preset for this architecture. If all the cascaded layers have the same function, then this is copied to '__activation_h_list'. 
      self.__activation_op = None;        # Activation of the output layer.
      self.__max_layers = None; # Holds the number of cascaded layers in the architecture. Determined by the learning algorithm. Not used. Controlled by the maxepoch of outer loop.
      self.__c_nhid_units = list ();      # TODO: List holding the number of hidden unit per cascaded layer. To be populated by the learning algorithm.
      self.__ip_op = None;  # HACK: Quick storage of input and output
      self.__cascade_node_types = list ();
    
    pass;
  
  
  # WORK IN PROGRESS, WARNING
  def copy_from (self, arch):
    
    self.__op_f = deepcopy (arch.get_op_final ());
    self.__op_f_unthresh = deepcopy (arch.get_op_unthresh_final ());
    self.__op_h = deepcopy (arch.get_op_hidden_layer ());
    self.__op_h_unthresh = deepcopy (arch.get_op_unthresh_hidden_layer ()); 
    self.__cW_h = deepcopy (arch.get_per_layer_cascade_weight ());
    self.__cW_o = deepcopy (arch.get_op_layer_weight ()); # A matrix. Once, __op_h is fully computed, this will hold the final matrix which does the final layer pass. This is grown by the training algorithm.
    self.__activation_h_list = arch.get_per_layer_cascade_activation (); #FIXME: Need a copy # List of activations at each cascaded layer, objects of 'activation_c' class. This can be populated by the learning algorithm.
    self.__activation_h_preset = arch.get_activation_h_preset (); #FIXME: Need a copy  # This is the type of activation function, an object of 'activation_c' which is preset for this architecture. If all the cascaded layers have the same function, then this is copied to '__activation_h_list'. 
    self.__activation_op = self.get_output_activation () #FIXME: Need a copy;        # Activation of the output layer.
    self.__max_layers = deepcopy (arch.get_max_layers ()); # Holds the number of cascaded layers in the architecture. Determined by the learning algorithm. Not used. Controlled by the maxepoch of outer loop.
    self.__c_nhid_units = deepcopy (arch.get_cascade_nhid_units ());      # TODO: List holding the number of hidden unit per cascaded layer. To be populated by the learning algorithm.
    self.__ip_op = deepcopy (arch.__get_ip_op ());  # HACK: Quick storage of input and output
    
  
  def do_forward_pass (self, X):
    
    ip_curr = X;
    
    self.__op_h = list ();
    self.__op_h_unthresh = list ();
    
    self.__op_h.append (ip_curr.copy ());
    self.__op_h_unthresh.append (ip_curr.copy ());
    
    for this_cascade_idx in range (len (self.__c_nhid_units)):
      
      # FIXME: Here we need the get_bias () function.
      op_this_h_unthresh = np.matmul (np.insert (ip_curr, ip_curr.shape[1], np.array ([1] * ip_curr.shape[0]), axis = 1), self.__cW_h[this_cascade_idx]);
      op_this_h = self.__activation_h_list[this_cascade_idx].act (op_this_h_unthresh);
                                                                                    
      # NOTE: We might get rid of storing the ip_curr as, it is essentially [X, self.__op_h] 

      #self.__op_h_unthresh = np.hstack ([self.__op_h_unthresh, op_this_h_unthresh]);
      #self.__op_h = np.hstack ([self.__op_h, op_this_h]);
      self.__op_h_unthresh.append (op_this_h_unthresh);
      self.__op_h.append (op_this_h);
      ip_curr = np.hstack ([ip_curr, op_this_h]); # Thresholded
    
    
    # Now we have all the hidden unit outputs.
    self.__op_f_unthresh = np.matmul (np.insert (ip_curr, ip_curr.shape[1], np.array ([1] * ip_curr.shape[0]), axis = 1), self.__cW_o);
    self.__op_f = self.__activation_op.act (self.__op_f_unthresh);

  def lazy_do_forward_pass (self):
    # TODO: Think about this feature where if we call this, it will just update with the
    # last cascaded layer. Therefore, avoiding computing the entire feed forward cycle and 
    # just append with the last one.
    
    pass;


  def predict (self, X):
    
    self.do_forward_pass (X);
    return self.__op_f; # Return reference. Faster.
  
  
  # Set input and output units (initial), maximum number of layers this architecture allows, the cascaded hidden layer activation functions, and the output activation
  def set_layers (self, ip_op, max_layers, activation_h, activation_op, W_op_init = None):
    
    self.__ip_op = ip_op.copy ();
    
    if (W_op_init is None):
      self.__cW_o = np.random.rand (ip_op[0] + 1, ip_op[1]);
    else:
      # TODO: Need dim validation
      self.__cW_o = W_op_init;
      
    self.__max_layers = max_layers;
    
    assert ((type (activation_h) is list) and (len (list (map (isinstance(activation_h, activation.activation_abs_c)))) == len (activation_h))) or (isinstance(activation_h, activation.activation_abs_c) == True), "Argument 'activation_h' shall be an object of subclass 'activation.activation_abs_c' or a list of 'activation.activation_abs_c'";
    
    if (isinstance (activation_h, activation.activation_abs_c)):
      self.__activation_h_preset = list ([activation_h]);
    else:
      self.__activation_h_preset = activation_h;
      
    self.__activation_op = activation_op;
    
    
  # TODO: Not yet decided what other feaures
  def init_weights (self, W_init_func = None):
    
    if (W_init_func is None):
      self.__W_init_func = np.random.rand;
    else:
      assert callable (W_init_func) or (type (W_init_func) is np.ndarray), "Argument 'W_init_func' should be a callable, list of numpy ndarray";
      self.__W_init_func = W_init_func;
      
    if (type (W_init_func) is np.ndarray):
      assert False, "NOT IMPLEMENTED: weight initialisation using an np.ndarray";
      pass; # TODO
      
    elif (callable (self.__W_init_func) is True):
      self.__cW_o = None;
      mrows, mcols = self.__ip_op[0] + 1, self.__ip_op[1];
      self.__cW_o = self.__W_init_func (mrows, mcols);
    else:
      pass;
    
  
  def get_op_layer_weight (self):
    return self.__cW_o;
  
  def get_max_layers (self):
    return self.__max_layers;
  
  def get_per_layer_cascade_weight (self):
    return self.__cW_h;
  
  def get_output_activation (self):
    return self.__activation_op;
  
  def get_per_layer_cascade_activation (self):
    return self.__activation_h_list;
  
  def get_op_hidden_layer (self):
    return self.__op_h;
  
  def get_op_final (self):
    return self.__op_f;
  
  def get_op_unthresh_hidden_layer (self):
    return self.__op_h_unthresh;
  
  def get_op_unthresh_final (self):
    return self.__op_f_unthresh;
  
  def get_cascade_nhid_units (self):
    return self.__c_nhid_units;
  
  def get_activation_h_preset (self):
    return self.__activation_h_preset;
  
  def set_op_layer_weight (self, cW_o):
    self.__cW_o = cW_o;
    
  def set_per_layer_cascade_weight (self, cW_h):
    self.__cW_h = cW_h;
  
  def set_per_layer_cascade_activation (self, ca_act):
    self.__activation_h_list = ca_act;
    
  def set_cascade_nhid_units (self, c_nhid_units):
    self.__c_nhid_units = c_nhid_units;
  
  def __get_ip_op (self):
    return self.__ip_op;
  
  def get_architecture (self):
    return [self.__ip_op[0], *self.get_cascade_nhid_units(), self.__ip_op[1]];
  
  def get_layer_types (self):
    return self.__cascade_node_types;
  
  def set_this_layer_type (self, node_type):
    # TODO: Assertion needed
    self.__cascade_node_types.append (node_type);
  
  # Overriding
  def get_weights (self):
    return np.concatenate ([np.concatenate (list (map (lambda x: x.flatten (), self.__cW_h))), self.__cW_o.flatten ()]);


  # TODO: Reset newtork function
  
