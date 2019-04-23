# NOTE: Does not work properly
class sgd_quickprop_c (sgd_learner_base_c):
  
  __update = None;
  __delW_old = None;
  
  def __init__ (self, nn_arch_obj, train_params, cost):
    super ().__init__ (nn_arch_obj, train_params, cost);
    self.__update = [np.ones (x.shape) for x in nn_arch_obj.get_per_layer_weight ()];
    self.__delW_old = [np.ones (x.shape) * 0.01 for x in self._delW];
    
  def _update_weights (self, W):
    for this_layer_idx in range (len (W)):
      
      this_update = self.__update[this_layer_idx] * (self._delW[this_layer_idx].T / (self.__delW_old[this_layer_idx].T - self._delW[this_layer_idx].T))
      self.__update[this_layer_idx] = this_update.copy ();
      W[this_layer_idx] = W[this_layer_idx] - self._params.get_param ("gamma") * self.__update[this_layer_idx];
      self.__delW_old[this_layer_idx] = self._delW[this_layer_idx].copy ();
      
  

