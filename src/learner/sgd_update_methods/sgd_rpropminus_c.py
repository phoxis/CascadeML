# NOTE: RProp+ implementation, looks like working. If yes, implement the others.
class sgd_rpropplus_c (sgd_learner_base_c):
  
  __delta  = None;
  __dW     = None;
  __eta_plus  = None;
  __eta_minus = None;
  __del_min   = None;
  __del_max   = None;
  __delW_old  = None;
  
  def __init__ (self, nn_arch_obj, train_params, cost):
    
    super ().__init__ (nn_arch_obj, train_params, cost);
    self.__delta  = [np.zeros_like (w) + np.array (0.0125) for w in nn_arch_obj.get_per_layer_weight ()];
    self.__dW = [np.zeros_like (w) for w in nn_arch_obj.get_per_layer_weight ()];
    self.__delW_old = [np.ones (x.shape) * 0.01 for x in self._delW];
    
    self.__eta_plus  = self._params.get_param ("rprop.etaplus");
    self.__eta_minus = self._params.get_param ("rprop.etaminus");
    self.__del_min   = self._params.get_param ("rprop.delmin");
    self.__del_max   = self._params.get_param ("rprop.delmax");
    
    if (self.__eta_plus == None):
      self.__eta_plus = 1.2;
    
    if (self.__eta_minus == None):
      self.__eta_minus = 0.5;
      
    if (self.__del_min == None):
      self.__del_min = 0.0;
      
    if (self.__del_max == None):
      self.__del_max = 50.0;
    
  def _update_weights (self, W):
    
    for this_layer_idx in range (len (W)):
      sgn_mat = np.sign (self.__delW_old[this_layer_idx] * self._delW[this_layer_idx]);
      sgn_mat = sgn_mat.T;
      
      self.__delta[this_layer_idx][sgn_mat == +1] *= self.__eta_plus;
      self.__delta[this_layer_idx][sgn_mat == +1][self.__delta[this_layer_idx][sgn_mat == +1] > self.__del_max] = self.__del_max;
      
      self.__delta[this_layer_idx][sgn_mat == -1] *= self.__eta_minus;
      self.__delta[this_layer_idx][sgn_mat == -1][self.__delta[this_layer_idx][sgn_mat == -1] < self.__del_min] = self.__del_min;
      
      self.__dW[this_layer_idx] = np.sign (self._delW[this_layer_idx].T) * self.__delta[this_layer_idx];
      W[this_layer_idx] -= self.__dW[this_layer_idx];

      self.__delW_old[this_layer_idx] = self._delW[this_layer_idx].copy ();
    
