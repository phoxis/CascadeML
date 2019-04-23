from cost.cost_abs_c import cost_abs_c as cost_abs_c
import cost.functions

class cost_factory_c (cost_abs_c):
    
    @staticmethod
    def get_cost (cost_name):
      
      if (cost_name == "mse"):
        return cost.functions.mse_cost_c ();
    
      if (cost_name == "ccnn_cost"):
        return cost.functions.ccnn_cost_c ();
    
      if (cost_name == "bpmll_cost"):
        return cost.functions.bpmll_cost_c ();
      
      else:
        return None;
