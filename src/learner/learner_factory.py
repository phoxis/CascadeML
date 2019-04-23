class learner_factory_c (nn_learner_abs_c):
    
    @staticmethod
    def get_learner (learner_name, arch, param, cost):
      
      if (learner_name == "sgd"):
        return sgd_with_momentum_c (arch, param, cost);
      
      if (learner_name == "rmsprop"):
        return sgd_rmsprop_c (arch, param, cost);
      
      if (learner_name == "rprop"):
        return sgd_rprop_c (arch, param, cost);
      
      if (learner_name == "quickprop"):
        return sgd_quickprop_c (arch, param, cost);

      else:
        return None;
