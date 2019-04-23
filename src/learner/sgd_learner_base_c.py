# FIXME: This package needs redesin, need to get the gradients from the architecture, we should not bother about things here. 
# MAJOR architecture change needed in the second revision

class sgd_learner_base_c (nn_learner_abs_c):
  
  def __init__ (self, nn_arch_obj, train_params, cost):
    super ().__init__ (nn_arch_obj, train_params, cost);
    self._nn_arch_obj.set_dropout_prob (train_params.get_param ("dropout"));
    
    self.__gradchkarch = None;
    # Initialise _delW.
    self._delW = self.__init_error_grad (self._nn_arch_obj.get_per_layer_units ());
    if (self._params.get_param ("gradcheck") is True):
      self.__gradchkarch = nn_ff_arch_c ();
      self.__gradchkarch.set_layers (self._nn_arch_obj.get_per_layer_units (), self._nn_arch_obj.get_per_layer_activation ());
      
    pass;
  
  def sgd_batch_train (self, train_data, batch_idx):
    
    # These variables are local.
    op_layers = None; # Holds the layer wise output of the network.
    
    D         = None;   # Holds the threshold derivative.
    e         = None;   # Holds the output layer error derivative.
    delta     = None;   # Holds the backpropagated layer-wise errors.
    
    activation_list = self._nn_arch_obj.get_per_layer_activation ();
    W               = self._nn_arch_obj.get_per_layer_weight     ();     # Get the reference list of per layer weight matrices
    
    #self.__clip_weight_norm (W); # WARNING
    
    nhid_units = self._nn_arch_obj.get_per_layer_units ();
    
    self._delW     = self.__init_error_grad (nhid_units);
    
    # Forward propagate
    self._nn_arch_obj.do_forward_pass (train_data["X"][batch_idx,]);
    
    op_final             = self._nn_arch_obj.get_per_layer_output ()[-1];
    op_layers_unthresh   = self._nn_arch_obj.get_per_layer_unthresh_output();
    op_layers            = self._nn_arch_obj.get_per_layer_output();
    dropout_mask         = self._nn_arch_obj.get_dropout_mask () if self._params.get_param ("dropout") is not None else None;
    
    # Backward propagate
    D           = self.__get_thresh_derivative           (op_layers_unthresh, activation_list, dropout_mask);
    e           = self.__get_op_error                    (op_final, train_data["y"][batch_idx,], self._cost);
    self._pre_update_weights                             (W);
    delta       = self.__back_propagate_error            (D, W, e);
    self.__update_error_grad                             (op_layers, delta);
    
    #self.__clip_grad_norm  (); #WARNING
    
    if (self._params.get_param ("wtreg.mode") in ["l1", "l2", "L1", "L2"]):
      self.__weight_regularisation                       (W, self._params.get_param ("wtreg.mode"), self._params.get_param ("wtreg.decay")/op_layers[0].shape[0]);
    
    # TODO: Remove from hereTesting numerical gradient
    grad_diff = 0;
    if (self._params.get_param ("gradcheck") is True):
      num_delW = self.__numeric_gradient (train_data["X"], train_data["y"], batch_idx, W, self._cost);
      grad = np.array ([]);
      num_grad = np.array ([]);
      for this_layer_idx in range (len (W)):
        grad = np.insert (grad, len (grad), self._delW[this_layer_idx].flatten ());
        num_grad = np.insert (num_grad, len (num_grad), num_delW[this_layer_idx].flatten ());
      #print (num_delW);
      #print (self._delW);
      # NOTE: Now numerical gradients work. adjustments done for specific cost function MSE
      #grad_diff += np.linalg.norm (2 * grad / (len (batch_idx) * y.shape[1]) - num_grad) / max (np.linalg.norm (num_grad), np.linalg.norm (grad)); # If we *do not* divide gradient with nos of examples
      grad_diff += np.linalg.norm (2 * grad / (y.shape[1]) - num_grad) / max (np.linalg.norm (num_grad), np.linalg.norm (grad)); # If we divide gradient with nos of examples
      #print (round (np.mean (diff_delW[this_layer_idx]), 6))
      print (round (grad_diff, 6))
    
    # Update weights for this batch
    self._update_weights (W); # A protected virtual function to be overridden by a subclass, who can implement rprop, qprop etc.
    self._nn_arch_obj.set_per_layer_weight (W);
    
  # TODO: Ongoing. Should not be here in this function
  def __numeric_gradient (self, X, y, batch_idx,  W, cost):
    
    epsilon = 10e-5;
    W_copy = [w.copy () for w in W];
    pt1_mat = [np.zeros_like (w) for w in W];
    pt2_mat = [np.zeros_like (w) for w in W];
    numgrad = [np.zeros_like (w) for w in W];
    
    for this_layer_idx in range (len (W)):
      for this_row in range (W[this_layer_idx].shape[0]):
        for this_col in range (W[this_layer_idx].shape[1]):

          W_copy[this_layer_idx][this_row][this_col] += epsilon;
          self.__gradchkarch.set_layers (self._nn_arch_obj.get_per_layer_units (), self._nn_arch_obj.get_per_layer_activation (), W_copy);
          pt2_mat[this_layer_idx][this_row][this_col] = cost.cost (self.__gradchkarch.predict (X[batch_idx,]), y[batch_idx,]);# + (0.5 * self._params.get_param ("wtreg.decay") * np.sum (np.square (np.concatenate (list (map (lambda x: x.flatten (), W_copy))))));
          
          W_copy[this_layer_idx][this_row][this_col] += (-2*epsilon);
          self.__gradchkarch.set_layers (self._nn_arch_obj.get_per_layer_units (), self._nn_arch_obj.get_per_layer_activation (), W_copy);
          pt1_mat[this_layer_idx][this_row][this_col] = cost.cost (self.__gradchkarch.predict (X[batch_idx,]), y[batch_idx,]);# + (0.5 * self._params.get_param ("wtreg.decay") * np.sum (np.square (np.concatenate (list (map (lambda x: x.flatten (), W_copy))))));
    
          W_copy[this_layer_idx][this_row][this_col] += epsilon; # Reset to original
          #if not np.isclose (self._delW[this_layer_idx][this_row][this_col], (pt2_mat[this_layer_idx][this_row][this_col] - pt1_mat[this_layer_idx][this_row][this_col])/ (2 * epsilon)):
            #print ("NO");   
      
      numgrad[this_layer_idx] = (pt2_mat[this_layer_idx] - pt1_mat[this_layer_idx]).transpose () / (2 * epsilon);
    
    return (numgrad)
    
    
  def __init_error_grad (self, nhid_units):
    
    delW  = list (); # Local
    for this_layer_idx in range (len (nhid_units) - 1):
      delW.append (np.zeros ([nhid_units[this_layer_idx] + 1, nhid_units[this_layer_idx + 1]]).transpose ());
  
    return delW;
      
      
  def __get_thresh_derivative (self, op_layers_unthresh, activation_obj_list, dropout_mask):
    
    D = list (); # Local # Length total layers - 1
    
    for this_layer_idx in range (len (op_layers_unthresh) - 1):
      # Each D[i] is the a matrix of the column vectors of the threshold derivatives for each datapoint at the t^th layer
      
      D.append ((activation_obj_list[this_layer_idx].act_dash (op_layers_unthresh[this_layer_idx + 1]))[:,:].transpose ());
      if (dropout_mask is not None):
        D[this_layer_idx] = (D[this_layer_idx].T * dropout_mask[this_layer_idx][-1]).T;
   
    return D;
  
  
  def __get_op_error (self, op_final, target_val, cost_obj):
    # Return a matrix each columns are the error vector for the output layer.
    return cost_obj.cost_dash (op_final, target_val).transpose ();
  
  @classmethod
  def __back_propagate_error (self, D, W, e):
    
    delta = [None] * (len (W)); # Local variable
    delta[len(W) - 1] = D[len (W) - 1] *  e;

    for this_layer_idx in range (len (W) - 2, -1, -1):
      delta[this_layer_idx] = D[this_layer_idx] * np.matmul (W[this_layer_idx + 1][range (W[this_layer_idx + 1].shape[0] - 1),], delta[this_layer_idx + 1]);
      
    return delta;
      
  
  
  def __update_error_grad (self, op_layers, delta):
    
    # Get number of examples
    nos_example = op_layers[len (op_layers) - 1].shape[0];
    
    for this_layer_idx in range (len (delta)):
      this_bias = self._nn_arch_obj.get_bias ()[this_layer_idx];
      self._delW[this_layer_idx] = np.matmul (delta[this_layer_idx], np.insert (op_layers[this_layer_idx], op_layers[this_layer_idx].shape[1], np.array (this_bias), axis = 1));
      #Shall we divide?
      self._delW[this_layer_idx] = self._delW[this_layer_idx] / nos_example;
      

  # NOTE: Can be done in a subclass or just here instead of increasing the class count unnecessarily
  def __weight_regularisation (self, W, wtreg_mode_p, coef_val):
    
    for this_layer_idx in range (len (W)):
      self._delW[this_layer_idx][:,range(0,W[this_layer_idx].shape[0]-1)] += coef_val * W[this_layer_idx][range (0,W[this_layer_idx].shape[0]-1),:].T;
      #self._delW[this_layer_idx] += coef_val * W[this_layer_idx].transpose ();
      
  # WARNING: WORK IN PROGRESS
  def __clip_weight_norm  (self, W):
    for this_layer_idx in range (len (W)):
      norms = np.sqrt (np.sum (np.power (W[this_layer_idx], 2), axis = 0));
      #print (W[this_layer_idx].shape);
      #print (norms.shape);
      #print (norms);
      # NOTE: A column in the matrix is a neuron.
      # FIXME: IF the below is correct, this needs to be very efficient
      limit = 4;
      norms[norms <= limit] = limit - 10e-10;
      W[this_layer_idx] = (W[this_layer_idx] * (limit / (10e-10 + norms)));
      #norms = np.sqrt (np.sum (np.power (W[this_layer_idx], 2), axis = 0));
      #print (W[this_layer_idx])
  
  
  # WARNING: WORK IN PROGRESS
  def __clip_grad_norm  (self):
    for this_layer_idx in range (len (self._delW)):
      norms = np.sqrt (np.sum (np.power (self._delW[this_layer_idx], 2), axis = 0));
      #print (W[this_layer_idx].shape);
      #print (norms.shape);
      #print (norms);
      # NOTE: A column in the matrix is a neuron.
      # FIXME: IF the below is correct, this needs to be very efficient
      limit = 1;
      norms[norms <= limit] = limit - 10e-10;
      self._delW[this_layer_idx] = (self._delW[this_layer_idx] * (limit / (10e-10 + norms)));
      #norms = np.sqrt (np.sum (np.power (W[this_layer_idx], 2), axis = 0));
      #print (W[this_layer_idx])
      
      
  # Overriding
  def train_batch (self, train_data, batch_idx):
    
    # assert that all the required params are present in the train_params for this algorithm to run
    # Although this will require us to know rprop and etc type. Do any conversion etc needed.
    # I think we need to initialise the initial weights here, somehow (instead of in the architecture)
    self.sgd_batch_train (train_data, batch_idx);
    
    pass;
  
  
  # NOTE: This is also a method which the subclass may override. This can be specially helpful 
  # when updating the weights based on some heuristics before finding the gradients. If an override
  # is not done, then it does nothing. Can be helpful to implement Nesterov momentum and other 
  # methods to update weights which finds the derivative at another point.
  def _pre_update_weights (self, W):
    pass;
  
  # NOTE: Should accept the reference of list of weights, then it can access the other information from the object of its superclass and update the W reference. Protected method.
  @abc.abstractmethod
  def _update_weights (self, W):
    pass;
  
