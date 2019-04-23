import numpy as np
import itertools
from .. import cost_abs_c as cost_abs_c

# TODO: Weight the pairing of BPMLL cost function based on the correlation or association between two labels.
# Or, we can sample from the set of labels based on the associations to select the positive and negative sets so that the number of pairwise computation done minimises while computing every iteration.
# 
class bpmll_cost_c (cost_abs_c):
  
  def __init__ (self):
    super ().__init__ ();
    self.__eps = 0;
  
  def cost_func (self, pred, target, args = None):
    
    n = pred.shape[0];
    total_err = 0;
    for this_idx in range (n):
      y_pos_labs = np.where (target[this_idx,:] == +1)[0];
      y_neg_labs = np.where (target[this_idx,:] == -1)[0];
    
      if ((len (y_pos_labs) is 0) or (len (y_neg_labs) is 0)):
        continue;
        
      total_err += np.sum ([np.exp ( -(pred[this_idx][yi] - pred[this_idx][yidash])) for yi, yidash in itertools.product (y_pos_labs, y_neg_labs)]) / (len (y_pos_labs) * len (y_neg_labs)  + self.__eps);
    
    # The second term to reduce the base cost, so that now the minima will be zero.
    return total_err / n;# - np.exp (-2) / n;


  def cost_dash_func (self, pred, target, args = None):
    
    n = pred.shape[0];
    p = pred.shape[1];
    
    dE = np.zeros_like (target, np.float64);
    
    for this_idx in range (n):
      y_pos_labs = list (np.where (target[this_idx,:] == +1)[0]);
      y_neg_labs = list (np.where (target[this_idx,:] == -1)[0]);
      
      if ((len (y_pos_labs) is 0) or (len (y_neg_labs) is 0)):
        continue;
        
      for j in range (p):
        this_error = 0;
        if (j in y_pos_labs):
          this_error = (+np.sum ([np.exp ( -(pred[this_idx,j]  - pred[this_idx,yidash])) for yidash in y_neg_labs]));
        else:
          this_error = (-np.sum ([np.exp ( -(pred[this_idx,yi] - pred[this_idx,j]))      for yi     in y_pos_labs]));
          
        dE[this_idx,j] = this_error / (len (y_pos_labs) * len (y_neg_labs) + self.__eps);
        
    return -dE;


    
