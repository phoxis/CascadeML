# WARNING: NOT TESTED NOT CORRECT

import numpy as np
from .. import cost_abs_c as cost_abs_c

class ccnn_cost_c (cost_abs_c):
  
  def __init__ (self):
    super ().__init__ ();
  
  @classmethod
  def cost_func (self, cerror, neterr, args = None):
    #print ("func", cerror.shape);
    #print ("func", neterr.shape);
    #print ("");
    #tmp =  np.sum (np.abs (np.sum ((cerror - np.mean (cerror, axis = 0))[:,:] * (neterr - np.mean (neterr, axis = 0)), axis = 1)));
    tmp =  np.sum (np.abs (np.sum ((cerror - np.mean (cerror, axis = 0))[:,:] * (neterr - np.mean (neterr, axis = 0)), axis = 0)));

    return tmp;

  # TODO: The args can have parameter which will be the mean error, and the sign of the correlation vector
  @classmethod
  def cost_dash_func (self, cerror, neterr, args = None):
    #print ("func_dash cerror", cerror.shape)
    #print ("func_dash neterr", neterr.shape)

    #tmp1 = (neterr - np.mean (neterr, axis = 0)); # n x o
    #tmp2 = np.sum ((cerror - np.mean (cerror, axis = 0))[:,np.newaxis] * tmp1, axis = 0); # n x 1 * n x o = n x o, column wise multiplication
    #tmp3 = -1 * np.sign (tmp2) * tmp1;
    #return (np.sum (tmp3, axis = 1));
    axis_order = list ([0, 1]);
    #print (np.sign (self.cost_func_by_axis (cerror, neterr, axis_order)))
    # FIXME:
    corsigns = np.sign (np.sum ((cerror - np.mean (cerror, axis = 0))[:,:] * (neterr - np.mean (neterr, axis = 0)), axis = 0));
    tmp1 = corsigns * (neterr - np.mean (neterr, axis = 0));
    #print (corsigns); # NOTE: This sometimes becomes all zero.
    #print (np.sum (tmp1, axis = 1, keepdims = True));
    return (np.sum (-tmp1, axis = 1, keepdims = True));

    
  # TODO: to implement if required
  @classmethod
  def cost_func_by_axis (self, cerror, neterr, axis_order, args = None):
    #print ( np.sum ((cerror - np.mean (cerror, axis = 0))[:,:] * (neterr - np.mean (neterr, axis = 0)), axis = 0) );
    return (cerror - np.mean (cerror, axis = axis_order[0]))[:,:] * (neterr - np.mean (neterr, axis = axis_order[0]));
  
