import abc
import inspect

"""
****************************
* Paramaters
****************************

[batch.genfunction] optional

This is either a 
  [*] list of lists, where the second level lists indicate the indices for the batch.
  [*] or a callable which generates the list of indices. The callable should accept to arguments. 
      
        def foo (tot_elem, batch_size_p)
      
      here the 'tot_elem' indicates the total elements in the dataset.
      and the 'batch_size_p' indicates the size of each batch.

      this function returns a list of lists, where the second level lists indicate the indices for the batch.
  [*] NoneType
        in which case mini batches will not be used for training and the entire training set will be used.
      

[batch.size] mandatory if 'batch.genfunction' is a callable

This is used if the parameter 'batch.genfunction' is a callable. Else it is unused

[maxepoch] mandatory

Maximum number of epochs

[retcost] optional

A flag indicating if to compute the training (and validation) costs or not.
If True, then the training process computes the costs and stores them.
If False, the cost computation is skipped.

[verbose] optional

Prints the training feedback

This is either
  [*] boolean, if True, then the default verbose message will be printed
  [*] Else, if it is a callable, then it expects a function pointerr which accepts a hash, 
      {'iter': present count of iterations,
       'max_iter': Maximum count of iterations,
       'train_err': The training error for current iteration,
       'valid_err': The validation error for current iteration
      }

[verbose.gran] optional

An integer indicating, if verbose is true, after how many training iteraations the feedback will be shown.

[wtfunc] optional

This is either a
  [*] Either a callable, a function which receives two dimensions and returns the initial weights
        
        def foo (nrow, ncol)
        
        nrow is the nrow x ncol determines dimension of a matrix. This will be called to populate the
        intial weight matrices.
  [*] A list of numpy.ndarray indicating the weights of the network, should be same as the architecture defined
  [*] NoneType
        'numpy.random' is used to generate the initial matrices

[min.error.train] optional

Indicates the training error to be reached before stopping.
If -1, then run until the iterations finish


[min.error.valid] optional

Indicates the training error to be reached before stopping.
If -1, then run until the iterations finish. This overrides min.error.train.

[earlystop.patience] optional

Indicates the patience window for early stopping. After every "patience" number of iterations, the present validation
error for the present epoch and the epoch "patience" iterations before is checked. Then the error the last best value
is restored. A value of None indicactes that early stopping is disabled

[earlystop.restore] optional

A boolean or none. Thiis indicates if the last best model within "earlystop.patience" window is to be restored while 
performing early stop or not. If False or None, then after performing early stop, the last model trained will be selected
else if True, then within the window of (self.epoch - earlystop.patience) to self.epoch, the best model based on validation
score will be selected and restored. If "earlystop.patience" is not enables, this argument does nothing.


[earlystop.type] optional

A string. Which can be
"window_edge" to compare the two edge values of the sliding window of validation cost with window width being the earlystop.patience
"moving_avg" to compare the current cost with the average of the last earlystop.patience validation scores

If None, then it is "window_edge" be default.

[stagnation.patience] optional

Indicates the patience windoe for stagnation detection. After every "stagnation.patiencce" number of iterations, the
present validation error for the present epoch "stagnation.patiencce" iterations before is checked. Then the error at 
the last best value is restored. A value of None indicates the stagnation stopping is disabled.

[stagnation.tolerance] optional

This is a real value indicating the absolute difference between the errors which shall be considered as stagnation.


[wtreg.mode] optional

This indicated the weight regularisation mode. Can be "l1" or "L1" and "l2" or "L2" for l1 and L2 regularisation respectively
If not present, no regularisation will be applied

[wtreg.decay]

If "wtreg.mode" is defined, then this used as the coefficient. Should be in the range of [0,1]. If not defined, default is 0.00001


"""
from sklearn.linear_model import LinearRegression
class nn_learner_abs_c (metaclass = abc.ABCMeta):
  
  
  
  def __init__ (self, nn_arch_obj, train_params, cost, algo_name = "sgd"): # algo_name will decide which class to instantise
    # Private
    self._params = None;            # NOTE: An instance of nn_parameters_c
    self.__name = "Unknown learner";    # Algo name
    
    
    # Protected
    self._nn_arch_obj = None;
    self._cost = None;            # A 'cost_abs_c' type
    
    self.__train_errors  = None;
    self.__valid_errors  = None;
    self.__metric_values = dict ({"train": {}, "valid": {}}); # Is a hash, of hashes. First level is "train" and "valid", then each metric name is a key which has a list of the values.
    self.epoch = 0;
    
    self.metric_functs = list (); # List of functions to apply for metrics. # TODO
  
    assert ((nn_arch_abs_c in inspect.getmro (type (nn_arch_obj))) == True), "Argument 'nn_arch_obj' should be an object of class 'nn_arch_abs_c's";
    # TODO: Assert X type and y type. Allow pandas dataframes
    # TODO: Assert train_params which are mandatory. Or get the optional params out of the hash train_params
    
    self._nn_arch_obj = nn_arch_obj;
    self._params      = nn_parameters_c (train_params);
    self._cost        = cost;

  # This class should not hold the train and validation dataset as it's member.
  # Validation dataset is for early stop and etc.
  # FIXME: THIS INTERFACE IS UGLY. Decide how the users should call this, either all arguments in this function and then call this without arguments or what.
  # train_data is a dict with two components "X", and "y".
  def train (self, train_data, nn_arch_obj = None, cost = None, validation_data = None, train_params = None):
    
    # Overwrite passed parameters on the existing. 'train_params' can be a dict or a type 'nn_parameters_c'
    self._params.update_params (train_params);
    
    # If batches are not set, make it on the full dataset.
    batch_gen_p  = self._params.get_param ("batch.genfunction");
    batch_size_p = self._params.get_param ("batch.size");
    if (batch_gen_p is None):
      batch_idx_list = list ([range (train_data["X"].shape[0])]);
    elif (type (batch_gen_p) is list):
      batch_idx_list = batch_gen_p;
    elif (callable (batch_gen_p) == True):
      assert (batch_size_p != None), "Parameter 'batch.size' is not provided. 'batch.size' is mandatory if 'batch.genfunction' is a callable";
      batch_idx_list = batch_gen_p (train_data["X"].shape[0], batch_size_p);
    else:
      # TODO: Exception
      pass;
    
    
    # _p for parameter.
    max_epoch_p                = self._params.get_param ("maxepoch");
    retcost_p                  = self._params.get_param ("retcost");
    wtfunc_p                   = self._params.get_param ("wtfunc");
    verbose_p                  = self._params.get_param ("verbose");
    verbose_gran_p             = self._params.get_param ("verbose.gran"); 
    min_error_train_p          = self._params.get_param ("min.error.train");
    min_error_valid_p          = self._params.get_param ("min.error.valid");
    early_stop_patience_p      = self._params.get_param ("earlystop.patience");
    early_stop_restore_p       = self._params.get_param ("earlystop.restore");
    early_stop_type_p          = self._params.get_param ("earlystop.type");
    stagnation_stop_patience_p = self._params.get_param ("stagnation.patience");
    stagnation_tolerance_p     = self._params.get_param ("stagnation.tolerance");
    compute_metrics_p          = self._params.get_param ("compute.metrics"); # A hash with keys "metric_name", "metric_value" and "threshold_func" (which may perfoms a thresholding). 
    wtreg_mode_p               = self._params.get_param ("wtreg.mode");  # NOTE: Does not belong here
    wtreg_decay_p              = self._params.get_param ("wtreg.decay"); # NOTE: Does not belong here
    
    valid_earlystop_types = ["window_edge", "moving_avg"];
    
    # FIXME: Get rid or retcost for computing the costs. Because we definitely need to compute different costs.
    retcost_p = True;
        
    if (stagnation_tolerance_p is None):
      stagnation_tolerance_p = 10e-8;
    
    self._nn_arch_obj.init_weights (wtfunc_p);
    
    assert (max_epoch_p != None), "Parameter 'maxepoch' is not set.";
    
    self.__train_errors = [None] * max_epoch_p;
    self.__valid_errors = [None] * max_epoch_p;
    
    if (min_error_train_p == None):
      min_error_train_p = -1;
    
    if (min_error_valid_p == None):
      min_error_valid_p = -1;
    
    # Stop based on validation set, if validation min error is set to some value
    if (min_error_valid_p != -1):
      min_error_train_p = -1;
      
    if (early_stop_type_p is None):
      early_stop_type_p = "window_edge";
      
    if (early_stop_type_p not in valid_earlystop_types):
      print ("Incorrect earlystop.type = \"", early_stop_type_p, "\". Should be in [", valid_earlystop_types, "]");
    
    # Will hold the older models
    nn_arch_restore_queue = list ();
    #for i in range (early_stop_patience_p):
      #nn_arch_restore_queue.append ({"arch": None, "train_cost": float ("inf"), "valid_cost": float ("inf")});
    
    early_stop_err_increase_streak = 0;
    
    metrics = dict ({"train": None, "valid": None});
    metric_name_list = list (map (lambda x: x["metric_name"], compute_metrics_p)) if (compute_metrics_p is not None) else [];
    for this_metric in metric_name_list:
      self.__metric_values["train"][this_metric] = [];
      self.__metric_values["valid"][this_metric] = [];
    
    for self.epoch in range (max_epoch_p):
      
      train_cost_val = 0;
      valid_cost_val = 0;
      if (validation_data is None):
        valid_cost_val = float ("inf");
      
      for this_batch_idx in batch_idx_list:
        
        # FIXME: Make sure if this makes any sense or not.
        self._nn_arch_obj.set_fixed_output_unthresh (self._nn_arch_obj.get_fixed_train_valid_op ("train"));

        # Train this batch
        self.train_batch (train_data, this_batch_idx);
        
        # FIXME: For CCNN, the cost updating is tricky. These costs can be a member of this class.
        # In that case the responsibility of cost updation is of the implementer's. OR the CCNN has 
        # to have different architecture.
        # Compute the cost if only the we want to return costs
        if (retcost_p == True):
          train_pred      = self._nn_arch_obj.predict (train_data["X"]);
          train_cost_val += self._cost.cost (train_pred, train_data["y"]);
          # NOTE: Implementing the weight regularisation thing here for now
              
          if (wtreg_mode_p in ["l2", "L2"]):
            train_cost_val += (wtreg_decay_p / (train_data["X"].shape[0] * 2)) * np.sum (self._nn_arch_obj.get_weights ()**2);
            #train_cost_val += wtreg_decay_p * np.sum (self._nn_arch_obj.get_weights ()**2);
                                                                          
        if (retcost_p == True): # OR we want to do early stopping based on this 
          if (validation_data != None):
            self._nn_arch_obj.set_fixed_output_unthresh (self._nn_arch_obj.get_fixed_train_valid_op ("valid"));
            valid_pred      = self._nn_arch_obj.predict (validation_data["X"]);
            valid_cost_val += self._cost.cost (valid_pred, validation_data["y"]);
            # NOTE: Implementing the weight regularisation thing here for now
            if (wtreg_mode_p in ["l2", "L2"]):
                valid_cost_val += (wtreg_decay_p / (train_data["X"].shape[0] * 2)) * np.sum (self._nn_arch_obj.get_weights ()**2);
                #valid_cost_val += wtreg_decay_p * np.sum (self._nn_arch_obj.get_weights ()**2);
            
        self._nn_arch_obj.set_fixed_output_unthresh (None);
      
      # Compute the other metrics after all the batches are complete
      if (retcost_p == True):
        if (compute_metrics_p is not None):
          train_pred       = self._nn_arch_obj.predict (train_data["X"]);
          metrics["train"] = list (map (lambda x: {"metric_name": x["metric_name"], "metric_value": x["metric_function"] (x["threshold_func"].predict_labels (train_pred), x["threshold_func"].predict_labels (train_data["y"]))}, compute_metrics_p));
          if (validation_data != None):
            valid_pred       = self._nn_arch_obj.predict (validation_data["X"]);
            metrics["valid"] = list (map (lambda x: {"metric_name": x["metric_name"], "metric_value": x["metric_function"] (x["threshold_func"].predict_labels (valid_pred), x["threshold_func"].predict_labels (validation_data["y"]))}, compute_metrics_p));

      
      if (retcost_p == True):
        self.__train_errors[self.epoch] = train_cost_val = train_cost_val / len (batch_idx_list);
        self.__valid_errors[self.epoch] = valid_cost_val = valid_cost_val / len (batch_idx_list);
        if (compute_metrics_p is not None):
          for this_metric_idx in range (len (metrics["train"])):
            self.__metric_values["train"][metrics["train"][this_metric_idx]["metric_name"]].append (metrics["train"][this_metric_idx]["metric_value"]);
            self.__metric_values["valid"][metrics["valid"][this_metric_idx]["metric_name"]].append (metrics["valid"][this_metric_idx]["metric_value"]);
      

      
      break_loop = False;
      epoch_trunc_idx = None;
      if (min_error_train_p != -1) and (train_cost_val <= min_error_train_p):
        if (verbose_p == True):
          print ("Min training error reached", train_cost_val);
        break_loop = True;
        epoch_trunc_idx = self.epoch + 1;
      
      
      if (min_error_valid_p != -1) and (valid_cost_val <= min_error_valid_p):
        if (verbose_p == True):
          print ("Min validation error reached");
        break_loop = True;
        epoch_trunc_idx = self.epoch + 1;
      
      # TODO: Fancy early stopping and stagnation detection and oscilation detection based on possibly trend detection by fitting linear model or some kind of running average etc
      if (early_stop_patience_p is not None) and (early_stop_restore_p is True):
        nn_arch_restore_queue.append ({"arch": type (self._nn_arch_obj)(self._nn_arch_obj), "train_cost": train_cost_val, "valid_cost": valid_cost_val});
        
      # DEBUG:
      #if (verbose_p is True):
        #print ("[[[[ ", self.epoch, " = ", self.__valid_errors[self.epoch], ",", self.epoch - early_stop_patience_p + 1, " = ", self.__valid_errors[self.epoch - early_stop_patience_p + 1], "]]]]");
        
      # Check every early_stop_patience_p iterations
      #if ((early_stop_patience_p is not None) and ((self.epoch + 1) % early_stop_patience_p == 0) and (self.__valid_errors[self.epoch] > self.__valid_errors[self.epoch - early_stop_patience_p + 1])):
      # Check every iteration
      #if ((early_stop_patience_p is not None) and (self.__valid_errors[self.epoch] > self.__valid_errors[max (0, self.epoch - early_stop_patience_p)])):
      # Check every iteration after minimum of early_stop_patience_p
      #print (early_stop_err_increase_streak)
      # NOTE: Add feature to moving median as well, assign function pointer here based on if-else and use it below?
      # TODO: Keep an actual running average instead of repeateadly doing an np.mean ?
      if ((early_stop_type_p is "window_edge") and ((early_stop_patience_p is not None) and (self.epoch > early_stop_patience_p) and (self.__valid_errors[self.epoch] > self.__valid_errors[self.epoch - early_stop_patience_p]))) or ((early_stop_type_p is "moving_avg") and ((early_stop_patience_p is not None) and (self.epoch > early_stop_patience_p) and (self.__valid_errors[self.epoch] > np.mean (self.__valid_errors[(self.epoch - early_stop_patience_p):(self.epoch-1)]))) ):
        # If the error increase streak is not long enough break, but count the decrease
        if (early_stop_type_p is "moving_avg") and (early_stop_err_increase_streak < early_stop_patience_p):
          early_stop_err_increase_streak += 1;
        elif (early_stop_type_p is not "moving_avg") or ((early_stop_type_p is "moving_avg") and (early_stop_err_increase_streak >= early_stop_patience_p)):
          if (verbose_p is True):
            print ("Early stopping based on ,\"", early_stop_type_p ,"\" method, on validation error.", end = "");
          break_loop = True;
          
          if ((early_stop_restore_p is None) or (early_stop_restore_p is False)):
            if (verbose_p is True):
              print ("Stopping training, using last model.");
              
            epoch_trunc_idx = max (1, (self.epoch - early_stop_patience_p + 1));
            
          if (early_stop_restore_p is True):
            if (verbose_p is True):
              print ("Stopping training, restoring the last best model.");
            
            # CLEANUP: Needed
            best_idx = -1;
            best_m = {"arch": None, "train_cost": float ("inf"), "valid_cost": float ("inf")};
            idx = 0;
            for this_m in nn_arch_restore_queue:
              if (this_m["valid_cost"] < best_m["valid_cost"]):
                best_m = this_m;
                best_idx = idx;
              
              # DEBUG:
              #print (np.round (this_m["train_cost"], 6), np.round (best_m["train_cost"], 6), np.round (this_m["valid_cost"], 6), np.round (best_m["valid_cost"], 6), idx);
              idx += 1;
            
            # DEBUG:
            #print ("best:", np.round (best_m["train_cost"], 6), np.round (best_m["valid_cost"], 6), best_idx);
            epoch_trunc_idx = max (1, (self.epoch - early_stop_patience_p  + best_idx + 1));
            train_cost_val = best_m["train_cost"];
            valid_cost_val = best_m["valid_cost"];
            # DEBUG:
            #print (np.round (self.__valid_errors[self.epoch], 6), "vs", np.round (self.__valid_errors[self.epoch - early_stop_patience_p + 1], 6), "at", self.epoch, "and", self.epoch - early_stop_patience_p + 1);
            #print (np.round (self.__train_errors[self.epoch], 6), "vs", np.round (self.__train_errors[self.epoch - early_stop_patience_p + 1], 6), "at", self.epoch, "and", self.epoch - early_stop_patience_p + 1);
            
            self._nn_arch_obj.copy_from (best_m["arch"]); # NOTE: Changing the original reference which was given
            
            # DEBUG:
            #valid_pred      = self._nn_arch_obj.predict (validation_data["X"]);
            #valid_cost_val  = self._cost.cost (valid_pred, validation_data["y"]);
            #if (valid_cost_val != best_m["valid_cost"]):
              #print ("Best model not restored properly");
            #print ("CONFIRM VALID COST (full batch mode): ", valid_cost_val);
      else:
        if (early_stop_type_p is "moving_avg"):
          early_stop_err_increase_streak = max (0, early_stop_err_increase_streak - 1);
        
      
      if (early_stop_patience_p is not None) and (early_stop_restore_p is True) and (self.epoch >= early_stop_patience_p):
          nn_arch_restore_queue.pop (0);  
      
      # FIXME: Bad stagnation algorithm. Detect using average or a continuous streak
      if ((stagnation_stop_patience_p is not None) and ((self.epoch + 1) % stagnation_stop_patience_p == 0) and (np.abs (self.__train_errors[self.epoch] - self.__train_errors[self.epoch - stagnation_stop_patience_p + 1]) < stagnation_tolerance_p)):
        if (verbose_p == True):
          print ("Stopping, stagnation detected");
        break_loop = True;
        epoch_trunc_idx = self.epoch;
      
      # NOTE: The default print can be put inside a function, but it is unnecessary.
      # The function pointer feature would allow the caller to put the results in a logger.
      # Also, if we want more thing to be logged then we can modify the hash.
      # TODO: Add verbose for also which batch is being executed.
      if (verbose_p == True):
        if (break_loop is True) or (verbose_gran_p is None) or (self.epoch % verbose_gran_p == 0) or (self.epoch == (max_epoch_p - 1)):
          if (validation_data == None):
            print ("Iter = %4d/%d (%6.2f%%) | Train cost = %10.6f" % ((self.epoch + 1), max_epoch_p, ((self.epoch + 1)/max_epoch_p) * 100, train_cost_val));
          else:
            print ("Iter = %4d/%d (%6.2f%%) | Train cost = %10.6f | Validation cost = %10.6f" % ((self.epoch + 1), max_epoch_p, ((self.epoch + 1)/max_epoch_p) * 100, train_cost_val, valid_cost_val), end = "");
            if (compute_metrics_p is not None):
              for metric_idx in range (len (compute_metrics_p)):
                print (", ", metrics["train"][metric_idx]["metric_name"], " train = ", metrics["train"][metric_idx]["metric_value"], end = "");
                print (", ", metrics["valid"][metric_idx]["metric_name"], " valid = ", metrics["valid"][metric_idx]["metric_value"], end = "");
          print ("");
            
      elif (callable (verbose_p) == True):
        if (break_loop is True) or (verbose_gran_p is None) or (self.epoch % verbose_gran_p == 0) or (self.epoch == (max_epoch_p - 1)):
          verbose_p ({"iter ": self.epoch, "max_iter": max_epoch_p, "train_err": train_cost_val, "valid_err": valid_cost_val, "metrics": metrics});
      
      if (break_loop is True):
        if (retcost_p == True):
          pass;
          del self.__train_errors[epoch_trunc_idx:max_epoch_p];
          del self.__valid_errors[epoch_trunc_idx:max_epoch_p];
        break;
    
  # FIXME: If accessed directly, this will return once less than actual
  def get_epoch (self):
    return self.epoch + 1; # Can be directly be accessed

  def get_errors (self):
    return {"train_error": self.__train_errors, "validation_error": self.__valid_errors, "metrics": self.__metric_values};
  
  def get_model  (self):
    return self._nn_arch_obj;

  # To override. Depending on the architecture and algorithmm this should learn for a given batch and update the
  # 'nn_arch_c' object's trainable variables.
  @abc.abstractmethod
  def train_batch (self):
    pass;
  
