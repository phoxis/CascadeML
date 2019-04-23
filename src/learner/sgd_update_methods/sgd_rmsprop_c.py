class sgd_rmsprop_c (sgd_learner_base_c):

  __delW_sqsum = None;
  __eps = None;
  
  def __init__ (self, nn_arch_obj, train_params, cost):
    super ().__init__ (nn_arch_obj, train_params, cost);
    self.__delW_sqsum = [np.zeros_like (x) for x in self._delW];
    self.__eps = self._params.get_param ("eps");
    if (self.__eps is None):
      self.__eps = 10e-10;
      
    assert (type (self.__eps) in [float, None]), "Parameter 'eps' must be 'float' type or None";

  # TESTING: Right now sometimes diverges
  def _update_weights (self, W):
     for this_layer_idx in range (len (W)):
      self.__delW_sqsum[this_layer_idx] = self._params.get_param ("alpha") * self.__delW_sqsum[this_layer_idx] + (1 - self._params.get_param ("alpha")) * np.square (self._delW[this_layer_idx]);
      #print ((self._params.get_param ("gamma") * self._delW[this_layer_idx] / (np.sqrt (self.__delW_sqsum[this_layer_idx]) + self.__eps)).transpose ());
      W[this_layer_idx] = W[this_layer_idx] - (self._params.get_param ("gamma") * self._delW[this_layer_idx] / (np.sqrt (self.__delW_sqsum[this_layer_idx]) + self.__eps)).transpose ();

