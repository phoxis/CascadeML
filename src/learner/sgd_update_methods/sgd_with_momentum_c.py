class sgd_with_momentum_c (sgd_learner_base_c):

  __vel = None;
  
  def __init__ (self, nn_arch_obj, train_params, cost):
    super ().__init__ (nn_arch_obj, train_params, cost);
    
    self.__vel = [np.zeros_like (x) for x in self._delW];
  
  # TESTING: Need to test the nesterov momentum
  def _pre_update_weights (self, W):
    
    if (self._params.get_param ("nesterov") is False):
      return;
    
    for this_layer_idx in range (len (W)):
      W[this_layer_idx] = W[this_layer_idx] + self._params.get_param ("alpha") * self.__vel[this_layer_idx].transpose ();
      

  # Overriding. NOTE: Need the params now. for gamma, alpha etc.
  def _update_weights (self, W):
    for this_layer_idx in range (len (W)):
      #self.__vel[this_layer_idx] = self._params.get_param ("gamma") * self._delW[this_layer_idx];
      self.__vel[this_layer_idx] = self._params.get_param ("alpha") * self.__vel[this_layer_idx] + self._params.get_param ("gamma") * self._delW[this_layer_idx];
      #self._delW[this_layer_idx] = -1 * self._params.get_param ("gamma") * self._delW[this_layer_idx] + self._params.get_param ("alpha") * self._delW_old[this_layer_idx];
    
    # Update weights
    for this_layer_idx in range (len (W)):
      W[this_layer_idx] = W[this_layer_idx] - self.__vel[this_layer_idx].transpose ();
      #W[this_layer_idx] = W[this_layer_idx] + self._delW[this_layer_idx].transpose ();
