from activation.activation_abs_c import activation_abs_c as activation_abs_c
import activation.functions

class activation_factory_c (activation_abs_c):
    
    @staticmethod
    def get_activation (act_name):
      
      if (act_name == "linear"):
        return activation.functions.linear_act_c ();
      
      if (act_name == "sigmoid"):
        return activation.functions.sigmoid_act_c ();
      
      if (act_name == "tanh"):
        return activation.functions.tanh_act_c ();
      
      if (act_name == "arctan"):
        return activation.functions.arctan_act_c ();
      
      if (act_name == "relu"):
        return activation.functions.relu_act_c ();
      
      if (act_name == "prelu"):
        return activation.functions.prelu_act_c ();
      
      if (act_name == "elu"):
        return activation.functions.elu_act_c ();
      
      if (act_name == "softplus"):
        return activation.functions.softplus_act_c ();
    
      else:
        return None;
