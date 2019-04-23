# A learner for cascade correlation neural network. Will train 'nn_ccnn_arch_c' type.
# Cascade correlation neural network architecture: https://papers.nips.cc/paper/207-the-cascade-correlation-learning-architecture.pdf
from random import randint
from joblib import Parallel, delayed
import dill
#import pathos.multiprocessing as multiprocessing
import multiprocessing
import itertools

# TODO: _train_candidate_layers and _train_output_layer can be made classes, of which the objects can be assigned as an aggregate of this class. Therefore, now for the different _train_candidate_layers and -_train_output_layer implementation it will be easier to plug in things.

class nn_ccnn_learner_base_c (nn_learner_abs_c):
  
  def __init__ (self, nn_arch_obj, train_params, cost_net, cost_cascade, output_layer_train_obj, candidate_unit_train_obj, param_op, param_cascade):

    super ().__init__ (nn_arch_obj, train_params, cost_net);
    
    
    # Protected variables
    self._cascade_cost  = cost_cascade; # The cost function to minimise at the cascade layer
    
    # NOTE: The below two references __op_layer_trainer_obj and __op_ffnn_temp and similar is needed as we do not allow to instantise the 'nn_learner_abs_c' without any argument, can be done later.
    # Private variables
    self.__op_layer_trainer_obj = output_layer_train_obj;      # An 'nn_learner_abs_c' reference of the class, not an object of the class, as we use this to instantise to train the output layer
    self.__cascade_layer_trainer_obj = candidate_unit_train_obj; # An 'nn_learner_abs_c' reference of the class, not an object of the class, as we use this to instantise to train the cascaded layers
    
    self.__op_ffnn_temp = None;         # Object returned after the 'nn_learner_abs_c' object is instantised
    #cascade_ffnn_temp = dict (); # List of references after the 'nn_learner_abs_c' object is instantised. Each key indicates the number of units. The value is an object of 'nn_learner_abs_c' type.
    
    self._params_op      = nn_parameters_c (param_op);   # Parameters for the output layer larner
    self._params_cascade = nn_parameters_c (param_cascade);   # Parameters for the cascade layer learner
    
    self.__params_op_tmp  = nn_parameters_c (self._params_op);   # Copy of 'self._params_op'. For internal use, as we intend to change some parameters (say, "wtfunc"), before calling the output layer trainer
    self.op_epochs = 0;
    self.candidate_epochs = list ();
    self.real_time_epochs = None;
    self.main_net_errors = dict ({"train_error": [], "validation_error": [], "metrics": {"train": {}, "valid": {}}});

    self.__ccnn_train_errors = list ();
    
    self.__op_h_valid   = None;
    self.__op_f_valid   = None;
    self.__target_valid = None;
    self.__X_valid      = None;
    self.__validation_flag_p = False;
    self.__op_f_unthresh_valid = None;
    
    self.__per_process_tasks = 1;

    pass;
  
  
  # Simple delta rule to update the output layers.
  # TODO NEXT: Make the main network multi-layer
  def _train_output_layer (self, ip_mat, target, W, cost): # Args, the inputs at the output layer (which is forward propagated by other layers), and the weights. Modify weights using reference.s
    self.__op_ffnn_temp.set_layers ([ip_mat.shape[1], target.shape[1]], self._nn_arch_obj.get_output_activation (), list ([W]));
    self.__op_ffnn_temp.set_bias ([1, 1]);
    self.__params_op_tmp.set_param ("wtfunc", list ([W])); # Start with the old weight
    trainer = self.__op_layer_trainer_obj (self.__op_ffnn_temp, self.__params_op_tmp, cost);
    
    validation_data = None;
    if (self.__validation_flag_p is True):
      validation_data = {"X": np.hstack (self.__op_h_valid), "y": self.__target_valid};
      
    trainer.train ({"X": ip_mat,"y": target}, train_params = None, validation_data = validation_data);
    
    self.op_epochs += trainer.get_epoch ();
    
    errors_main_net = trainer.get_errors ();
    self.main_net_errors["train_error"].append (errors_main_net["train_error"]);
    self.main_net_errors["validation_error"].append (errors_main_net["validation_error"]);
    
    for this_metric_name in errors_main_net["metrics"]["train"].keys ():
      if (this_metric_name not in self.main_net_errors["metrics"]["train"]):
        self.main_net_errors["metrics"]["train"][this_metric_name] = [];
        self.main_net_errors["metrics"]["valid"][this_metric_name] = [];
      self.main_net_errors["metrics"]["train"][this_metric_name].append (errors_main_net["metrics"]["train"][this_metric_name]);
      self.main_net_errors["metrics"]["valid"][this_metric_name].append (errors_main_net["metrics"]["valid"][this_metric_name]);
    
    W = self.__op_ffnn_temp.get_per_layer_weight ()[0];

    return W;
  
  
  # ip_mat         : The immediate input for every candidate layer
  # main_net_op    : The output of the existing network. This does not change before we add a new candidate unit
  # target         : The target variable for the dataset
  # cost_net       : The cost which is to be optimiseed for the network
  # cost_cascade   : The cost which is to be optimised for the cascaded layer
  # unit_range     : A 2dim list indicating the min and max nos of units to be considered. Use [1, 1] for one unit
  # act_types      : Activation types to consider when making the pool. Makes a pool per type of activation. A list of types of 'activation_abs_c'
  # max_candidates : Size of the pool per 'act_types'
  # expansion_type : An array indicating of what type of candidate units to be considered. Successor or sibling.

  # TODO NEXT: multi-label cascade. Feedback the learned labels as input in cascaded layer.
  
  # TODO: Implement: When adding a successor take the existing layer, and utilise its weights.
  # Args, Input which will be received by a candidate layer. Return, the new learned weights, which will then be added by the caller.
  def _train_candidate_layers (self, ip_mat_l_list, main_net_op, target, cost_net, cost_cascade, unit_range, act_types, act_output, max_candidates, method, expansion_type, sibling_bias):
    # Here for each candidate in pool, make an nn_ff_arch_c, and call the learner from the learner package
    #np.seterr(all = "raise");
    if (self._params.get_param ("verbose") == True):
      print ("Thread responsibility: (activation_types: ", list (map (lambda x: x.get_name (), act_types)), ", max_candidates: ", max_candidates, ", expansion_type: ", expansion_type, ")");

    cascade_ffnn_temp = dict ();
    if (method in ["cascor", "cascade2"]):
      #cost_net_cache = (main_net_op - target);
      cost_net_cache = cost_cascade.cost_dash_func (main_net_op, target);
    
    #cost_net_cache = (target - main_net_op);
    best_unit = {"wt_ip": None, "act": None, "units": None, "cost": None, "wt_op": None, "method": None, "type": None};
    this_proc_candidate_epopchs = list ();
    
    cmp_function = None;
    if (method == "cascor"):
      best_unit["cost"] = float (-1);
      cmp_function = lambda x, y: x > y;
      #sibling_bias = 1 + sibling_bias;
      
    elif (method in ["cascade2", "cascade3"]):
      best_unit["cost"] = float ("inf");
      cmp_function = lambda x, y: x < y;
      #sibling_bias = sibling_bias;
    
    wtfunc_p = self._params_cascade.get_param ("wtfunc");
    max_width_p = self._params_cascade.get_param ("max.width");
    if (max_width_p is None):
      max_width_p = ip_mat_l_list[0].shape[1] + self._params.get_param ("maxepoch") + 32; # Should be a function of the input
      
    assert (wtfunc_p is None) or callable (wtfunc_p) == True, "Parameter 'wtfunc' for candidate training should be a 'callable' or 'NoneType' for CCNN";
    if (wtfunc_p == None):
      wtfunc_p = np.random.rand;
    
    #self.candidate_epochs = list ();
    pool = dict ();
    ip_mat = None;
    ip_valid_mat = None;
    # For each act_types

    for this_act in act_types:
      # Make a pool
      pool[this_act.get_name ()] = dict ({"successor": None, "sibling": None});
      for candidate_type in expansion_type:
        pool[this_act.get_name ()][candidate_type] = [None] * max_candidates;
        # The first unit must a successor
        if (candidate_type == "sibling" and len (ip_mat_l_list) == 1):
          candidate_type = "successor";
          if (self._params.get_param ("verbose") == True):
            print ("First layer must be a successor");
          
          pool[this_act.get_name ()][candidate_type] = [None] * max_candidates;
        
        ## NOTE: Uncommenting this forces adding sibling only by skipping successors.
        #if (candidate_type == "successor" and len (ip_mat_l_list) > 1):
          #continue;
        
        if (self._params.get_param ("verbose") == True):
          print ("Training as: ", candidate_type, ", activation: ", this_act.get_name ());
          
        for i in range (max_candidates):
          if (candidate_type == "sibling"):
            #print ("sibling", end = ',');
            # NOTE: Need so consider skipping if this layer has enough units
            
            # TODO: To have hidden layer(s) in the main network, we need to hstack the hidden layer outputs here, that's it
            ip_mat = np.hstack (ip_mat_l_list[0:(len (ip_mat_l_list) - 1)]);
            if (self.__validation_flag_p is True):
              ip_valid_mat = np.hstack (self.__op_h_valid[0:(len (self.__op_h_valid) - 1)]);
          elif (candidate_type == "successor"):
              #print ("successor", end = ',');
              
            # TODO: To have hidden layer(s) in the main network, we need to hstack the hidden layer outputs here, that's it
            ip_mat = np.hstack (ip_mat_l_list[0:(len (ip_mat_l_list))]);
            if (self.__validation_flag_p is True):
              ip_valid_mat = np.hstack (self.__op_h_valid[0:(len (self.__op_h_valid))]);

          # Make the weight matrix based on the columns + 1 in ip_mat and the selected hidden layer size
          h_units = randint (unit_range[0], unit_range[1]);
          
          # Learn these selected weights by optimising cost_cascade
          if (h_units not in cascade_ffnn_temp):
            cascade_ffnn_temp[h_units] = nn_ff_arch_c ();
          
          trainer          = None;
          this_candidate_W = None; # candidate unit's input weights, cascor and cascade2
          this_op_W        = None; # candidate unit's output weights, cascade2
          
          validation_data = None;
          if (self.__validation_flag_p is True):
            if (method in ["cascor", "cascade2"]):
              cost_net_cache_valid = (self.__op_f_valid - self.__target_valid);
              validation_data = {"X": ip_valid_mat, "y": cost_net_cache_valid}
            elif (method in ["cascade3"]):
              validation_data = {"X": ip_valid_mat, "y": self.__target_valid}

            #cost_net_cache_valid = (self.__target_valid - self.__op_f_valid);
          
          
          # TODO: Partial label training.
          # Select a subset of labels and input variables, prepare the ip_mat and target as per that
          # Train cascade network independently
          # When the weights are returned, pad the weights using 0s corresponding to which the attributes were not selected.
          
          if (method == "cascor"):
            # CasCor
            cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units], this_act);
            trainer = self.__cascade_layer_trainer_obj (cascade_ffnn_temp[h_units], self._params_cascade, cost_cascade);
            trainer.train ({"X": ip_mat, "y": cost_net_cache}, train_params = None, validation_data = validation_data);
            this_candidate_W = cascade_ffnn_temp[h_units].get_per_layer_weight ()[0];
          
          if (method == "cascade2"):
            print ("CASCADE2")
            # Cascade 2
            # FIXME: Which one is correct
            #cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units, target.shape[1]], [*act_types, activation.functions.linear_act_c ()]);
            cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units, target.shape[1]], [this_act, activation.functions.linear_act_c ()]);
            #cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units, target.shape[1]], [*act_types, self._nn_arch_obj.get_output_activation ()]);
            cascade_ffnn_temp[h_units].init_weights (self._params_cascade.get_param ("wtfunc"));
            cascade_ffnn_temp[h_units].set_bias ([1, 0, 0]);
            
            #if (self._params.get_param ("wtmse") is True):
              ## TESTING: Weighting the costs per label
              #wt_vec = np.sum ((target - main_net_op) ** 2, axis = 0);
              #wt_vec = wt_vec / np.sum (wt_vec);
              #cost_cascade.set_key ("wts", wt_vec);
              #print (cost_cascade.get_key ("wts"))
              ## TESTING END
            
            trainer = self.__cascade_layer_trainer_obj (cascade_ffnn_temp[h_units], self._params_cascade, cost_cascade);
            trainer.train ({"X": ip_mat, "y": cost_net_cache}, train_params = None, validation_data = validation_data);
            this_candidate_W, this_op_W = cascade_ffnn_temp[h_units].get_per_layer_weight ();
            
          if (method == "cascade3"):
            print ("CASCADE3");
            #cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units, target.shape[1]], [*act_types, self._nn_arch_obj.get_output_activation ()]);
            cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units, target.shape[1]], [this_act, self._nn_arch_obj.get_output_activation ()]);
            #cascade_ffnn_temp[h_units].set_layers ([ip_mat.shape[1], h_units, target.shape[1]], [*act_types, activation.functions.linear_act_c ()]);
            cascade_ffnn_temp[h_units].init_weights (self._params_cascade.get_param ("wtfunc"));
            cascade_ffnn_temp[h_units].set_bias ([1, 0, 0]);
            
            cascade_ffnn_temp[h_units].set_fixed_train_valid_op (main_net_op, "train"); # NOTE: Should be unthresholded
            cascade_ffnn_temp[h_units].set_fixed_train_valid_op (self.__op_f_unthresh_valid, "valid"); # NOTE: Should be unthresholded
            
            #if (self._params.get_param ("wtmse") is True):
              ## TESTING: Weighting the costs per label, maybe average with the older values, maybe change with every iteration?
              #wt_vec = np.sum ((target - main_net_op) ** 2, axis = 0);
              #wt_vec = wt_vec / np.sum (wt_vec);
              #cost_cascade.set_key ("wts", wt_vec);
              #print (cost_cascade.get_key ("wts"))
              ## TESTING END
            
            trainer = self.__cascade_layer_trainer_obj (cascade_ffnn_temp[h_units], self._params_cascade, cost_cascade);
            trainer.train ({"X": ip_mat, "y": target}, train_params = None, validation_data = validation_data);
            this_candidate_W, this_op_W = cascade_ffnn_temp[h_units].get_per_layer_weight ();
            
            cascade_ffnn_temp[h_units].set_fixed_train_valid_op (None, "valid"); # NOTE: Reset to none
            cascade_ffnn_temp[h_units].set_fixed_train_valid_op (None, "train"); # NOTE: Reset to none
            
          
          this_proc_candidate_epopchs.append (trainer.get_epoch ());
          #print  (self.candidate_epochs)
            # WARNING; Override random weights for testing and validation only
            #this_candidate_W, this_op_W = np.random.uniform (-0.5, +0.5, this_candidate_W.shape), np.random.uniform (-0.5, +0.5, size = this_op_W.shape);
          
          errors = trainer.get_errors ();
          
          error_type = "train_error";
          if (self.__validation_flag_p is True):
            error_type = "validation_error";
          # NOTE: Maintaining a pool may not be required, wasteful.
          pool[this_act.get_name ()][candidate_type][i] = (this_candidate_W, errors[error_type][len (errors[error_type]) - 1]);

            
          this_error = errors[error_type][len (errors[error_type]) - 1];
          if (self._params.get_param ("verbose") == True):
            print ("Candidate error: ", this_error);
          
          # the sibling_bias stands for the fraction error to be added with respect to the existing error.
          # TESTING: Check if this formula is effective
          modified_error = this_error;
          if (candidate_type == "sibling"):
            if (method in ["cascade2", "cascade3"]):
              modified_error = this_error + sibling_bias * this_error;
            elif (method is "cascor"):
              modified_error = this_error - sibling_bias * this_error;
          
          if (cmp_function (modified_error,  best_unit["cost"]) == True): 
            best_unit["cost"]   = this_error;
            best_unit["wt_ip"]  = this_candidate_W.copy ();
            best_unit["act"]    = this_act;
            best_unit["units"]  = h_units;
            best_unit["type"]   = candidate_type;
            best_unit["method"] = method;
            if (method == "cascade2"):
              best_unit["wt_op"] = -this_op_W.copy (); # NOTE: Inverting the weights as per the algorithm
            elif (method == "cascade3"):
              best_unit["wt_op"] = this_op_W.copy ();
          
      if (self._params.get_param ("verbose") == True):
        print ("\nBest candidate cost: ", best_unit["cost"], ",", "units: ", best_unit["units"], ",", "candidate type: \"", best_unit["type"], "\",", "activation: \"", best_unit["act"].get_name (), "\"");
    
    # NOTE: This sends the list of all the epochs took in parallel to select the best one
    best_unit["epoch_list"] = this_proc_candidate_epopchs;
    return best_unit;
    # Return the learned weights OR the best one and the activation function as well.
    
    pass;
  
  
  
  def reset_learner (self):
    self._nn_arch_obj.init_weights (self._params_op.get_param ("wtfunc")); # NOTE: Weight reset here
    # Get cascade layer weight reference.
    ca_W = self._nn_arch_obj.get_per_layer_cascade_weight ();
    ca_W = list (); # NOTE: Weight reset here
    self._nn_arch_obj.set_per_layer_cascade_weight (ca_W);

    
    c_nhid_units = self._nn_arch_obj.get_cascade_nhid_units ();
    c_nhid_units = list ();
    self._nn_arch_obj.set_cascade_nhid_units (c_nhid_units);
  
  
  
  
  # FIXME: There are a lot of unnecessary forward passes here. Optimise them.
  def ccnn_train (self, train_data, batch_idx):
    
    valid_ccnn_methods = ["cascor", "cascade2", "cascade3"];
    
    ip_dim = train_data["X"].shape[1];
    op_dim = train_data["y"].shape[1];
    
    
    #print ("START");
    #a = nn_ff_arch_c();
    #a.set_layers ([train_data["X"].shape[1], train_data["y"].shape[1]], self._nn_arch_obj.get_output_activation ());
    #l = sgd_rprop_c(a, self.__params_op_tmp, self._cost);
    #l.train ({"X": train_data["X"],"y": train_data["y"]}, train_params = None, validation_data = {"X": self._params.get_param("ccnn.validationdata")["X"], "y": self._params.get_param("ccnn.validationdata")["y"]});
    #saved_wt = a.get_per_layer_weight ();
    #print ("END");
    
    self.__op_ffnn_temp = nn_ff_arch_c ();
    self.__op_ffnn_temp.set_layers ([ip_dim, op_dim], self._nn_arch_obj.get_output_activation ());

    # Get the output weight reference
    # NOTE: Reset the input and output weights ass this can be the next batch or iteration, which will not train except the dimensions are correct.
    # This will reconstruct the weights every epoch of the learning. We do not retrain the entire network here.
    #self.reset_learner ();
    
    self.__validation_flag_p = self._params.get_param("ccnn.validationdata") is not None;
    if (self.__validation_flag_p is True):
      self.__X_valid      = self._params.get_param("ccnn.validationdata")["X"];
      self.__target_valid = self._params.get_param("ccnn.validationdata")["y"];
    
    op_W = self._nn_arch_obj.get_op_layer_weight ();
    ca_W = self._nn_arch_obj.get_per_layer_cascade_weight ();
    
    self._nn_arch_obj.do_forward_pass (train_data["X"]);
    op_h = self._nn_arch_obj.get_op_hidden_layer ().copy (); #Copy as the next pass for validation will overwrite the reference
    op_f = self._nn_arch_obj.get_op_final ().copy ();
    op_unthresh_f = self._nn_arch_obj.get_op_unthresh_final().copy ();

    if (self.__validation_flag_p is True):
      self._nn_arch_obj.do_forward_pass (self.__X_valid);
      self.__op_h_valid = self._nn_arch_obj.get_op_hidden_layer ().copy ();
      self.__op_f_valid = self._nn_arch_obj.get_op_final ().copy ();
      self.__op_f_unthresh_valid = self._nn_arch_obj.get_op_unthresh_final().copy ();
    
    ca_act = self._nn_arch_obj.get_per_layer_cascade_activation ();
    c_nhid_units = self._nn_arch_obj.get_cascade_nhid_units ();

    # Train output layer only. updates op_W
    if (self.epoch == 0):
      # FIXME: Reset learner here?
      if (self._params.get_param ("verbose") is True):
        print ("Training output layer for the first epoch");
      op_W = self._train_output_layer (np.hstack (op_h), train_data["y"], op_W, self._cost);
      self._nn_arch_obj.set_op_layer_weight (op_W);
    
    # Get the initial weight initialiser function
    wtfunc_p = self._params_op.get_param ("wtfunc");
    parallel_p = self._params_cascade.get_param ("parallel");
    act_types_p = self._params_cascade.get_param ("act.types");
    
    if (act_types_p == None):
      act_types = [activation.activation_factory_c.get_activation("linear")]; # Default
    # TODO: Assert if we have correct format.
    
    # The cascade correlation method to follow
    ccnn_method = self._params.get_param ("ccnn.method"); # "cascor" or "cascade2".
    
    assert (ccnn_method in valid_ccnn_methods), "Parameter \"ccnn.method\" should be in [" + ",".join (valid_ccnn_methods) + "], found: " + ccnn_method;
    if (self._params.get_param ("verbose") == True):
      print ("Method:", ccnn_method);
    
    # Select if just cascade correlation or SDCC network
    valid_ccnn_expansion = ["sibling", "successor"];
    ccnn_expansion_type = self._params.get_param ("ccnn.expansion");
    if (ccnn_expansion_type == None):
      ccnn_expansion_type = valid_ccnn_expansion;
    ccnn_expansion_type = list (set (ccnn_expansion_type));
    
    assert (sum (list (map (lambda x: x in valid_ccnn_expansion, ccnn_expansion_type))) ==  len (ccnn_expansion_type)), "Parameter \"ccnn.expansion\" should be any of following or both: [" + ",".join (valid_ccnn_expansion) + "], provided is " + ",".join (ccnn_expansion_type);
    if (self._params.get_param ("verbose") == True):
      print ("Expansion method: [", ",".join (ccnn_expansion_type), "]");
    
    # Max candidate in pool
    ccnn_max_candidates = self._params.get_param ("ccnn.candidate.max");
    if (self._params.get_param ("verbose") == True):
      print ("Max candidates per type: ", ccnn_max_candidates);

    
    # Bias indicating if we should add more bias suucssor o chiildren
    ccnn_sibling_bias = self._params.get_param ("ccnn.sibling.bias");
    if (ccnn_sibling_bias == None):
      if (self._params.get_param ("verbose") == True):
        print ("Setting \"ccnn.sibling.bias\" to 0.0");
        
      ccnn_sibling_bias = 0.0;
    
    # The range of units using which the hidden cascacde layers will be added, except "cascade-correlation" algorithm. In that case this will be forced to 1.
    ccnn_chidunits_range = self._params.get_param ("ccnn.chidunits.range");
    if (ccnn_method == "cascor"):
      ccnn_chidunits_range = [1, 1];
      if (self._params.get_param ("verbose") == True):
        print ("Resetting \"ccnn.chidunits.range\" to [1, 1], as the \"method\" is selecte to be \"cascacde\"");
    
    # FIXME: Assertion should detect, if float, then the range should be (0,2], else [1, n]
    assert ( (type (ccnn_chidunits_range) is str) or ((ccnn_chidunits_range is not None) and (len (ccnn_chidunits_range) == 2) and (sum (list (map (lambda x: x > 0, ccnn_chidunits_range))) == len (ccnn_chidunits_range)) and (ccnn_chidunits_range[0] <= ccnn_chidunits_range[1]) ) ), "Parameter \"ccnn.chidunits.range\" should define a range of number of hidden units [x, y], where 1 <= x <= y, provided is [" + ",".join (map (lambda x: str (x), ccnn_chidunits_range)) + "]";
    
    assert (wtfunc_p is None) or callable (wtfunc_p) == True, "Parameter 'wtfunc' for the output layer training should be a 'callable' or 'NoneType' for CCNN";
    if (wtfunc_p == None):
      wtfunc_p = np.random.rand;
      
    assert ((parallel_p is None) or (type (parallel_p) is bool) or ((type (parallel_p) is int) and (parallel_p > 0 and parallel_p <= 32))), "argument \"parallel\" should be either None, a boolean, or a positive integer less than or equal to 32";
    if ((type (parallel_p) is bool) and (parallel_p is True)):
      parallel_p = multiprocessing.cpu_count ();
      
    parallel_obj = Parallel (n_jobs = parallel_p, backend = "threading");
    # START training
    #for this_cascade_idx in range (self._nn_arch_obj.get_max_layers()):
    if (self._params.get_param ("verbose") == True):
      print ("Training Cascade Layer " + str (self.epoch) + "\n");
    
    unit_selection_range = None;
    if ((type (ccnn_chidunits_range) is str) and (ccnn_chidunits_range == "auto")):
      unit = ip_dim + len (self._nn_arch_obj.get_cascade_nhid_units ()) + 1;
      unit_selection_range = [unit, unit];
    else:
      unit_selection_range = ccnn_chidunits_range.copy ();
      # Allowing maximum of twice the number of input units
      if (sum (list (map (lambda x: 0 < x <= 2 and type (x) is float, unit_selection_range))) == len (unit_selection_range)):
        if (self._params.get_param ("verbose") == True):
          print ("Hidden unit ranges in cascade layer selection are in fractions. Should be within (0.0, 2.0]");
        unit_selection_range = list (map (int, list (np.round (np.array (unit_selection_range) * ip_dim))));
      else:
        unit_selection_range = list (map (int, unit_selection_range));
        if (self._params.get_param ("verbose") == True):
          print ("Hidden units selection ranges are fixed");
      
    if (self._params.get_param ("verbose") == True):
      print ("Unit selection range: [", unit_selection_range[0], ", ", unit_selection_range[1], "]");
      
    # TESTING:
    if (ccnn_method == "cascade3"):
      main_net_op = op_unthresh_f;
      #self._cost_cascade = self._cost; # Optimise the same cost function
    else:
      main_net_op = op_f;
      

    this_layer_epochsets = list ();
    candidate_train_return = None;
    # TODO: Maybe a wrapper here to put the below two and the seletion code confined somewhere
    if (parallel_p is None) or  (parallel_p is False):
      # Make canididate cascade layer. op_h will be updated here (and extended), train_data can be used with _cascade_cost. Returns a list of candidate weights with their costs. Then we can decide the best from that.
      candidate_train_return = self._train_candidate_layers (op_h, 
                                                             main_net_op, 
                                                             train_data["y"], 
                                                             self._cost, 
                                                             self._cascade_cost, 
                                                             unit_selection_range, 
                                                             act_types_p, 
                                                             self._nn_arch_obj.get_output_activation (), 
                                                             ccnn_max_candidates, 
                                                             ccnn_method, 
                                                             ccnn_expansion_type, 
                                                             ccnn_sibling_bias);
      
      this_layer_epochsets.append (candidate_train_return["epoch_list"]);
    
    # Training candidate layer.
    if (parallel_p is not None) and (type (parallel_p) is int):
      self.__per_process_tasks = int (np.ceil (ccnn_max_candidates/parallel_p));
      tasks = min (parallel_p, ccnn_max_candidates) * len (ccnn_expansion_type);
      
      # TODO: Next, if possible keep an option to parallalise number of  units per type per activation
      candidate_train_return_list = parallel_obj (
                                                  delayed (self._train_candidate_layers) 
                                                  (op_h, 
                                                   main_net_op, 
                                                   train_data["y"], 
                                                   self._cost, 
                                                   self._cascade_cost, 
                                                   unit_selection_range, 
                                                   [this_act_type], 
                                                   self._nn_arch_obj.get_output_activation (), 
                                                   self.__per_process_tasks, 
                                                   ccnn_method, 
                                                   [ccnn_expansion_type[i % len(ccnn_expansion_type)]], 
                                                   ccnn_sibling_bias) for i, this_act_type in itertools.product (list (range (tasks)), act_types_p)
                                                 );
      
      
      candidate_train_return = candidate_train_return_list[0];
      
      
      # FIXME: Looks like this has a problem in performance
      # Get the best cost children from the pool returned. This is to handle the parallel function
      cmp_function = None;
      if (ccnn_method == "cascor"):
        cmp_function = lambda x, y: x > y;
      elif (ccnn_method in ["cascade2", "cascade3"]):
        cmp_function = lambda x, y: x < y;
      print ("ALL COSTS: ", list (map (lambda x: x["cost"], candidate_train_return_list)));
      best_cost = candidate_train_return["cost"];
      for this_candidate in candidate_train_return_list:
        this_layer_epochsets.append (this_candidate["epoch_list"]);
        if (cmp_function (this_candidate["cost"], best_cost) == True):
          candidate_train_return = this_candidate;
          best_cost = this_candidate["cost"];
      print ("BEST COST: ", best_cost);
    
    self.candidate_epochs.append (this_layer_epochsets);
    
    
    # Select from the candidate_list above, AND also the activation.
    if (candidate_train_return["type"] == "successor"):
      if (self._params.get_param ("verbose") == True):
        print ("Adding: SUCCESSOR")
      
      ## for TESTING only. This appends random weights.
      #ca_W.append         (wtfunc_p (candidate_train_return["wt_ip"].shape[0], candidate_train_return["wt_ip"].shape[1]));
      
      ca_W.append         (candidate_train_return["wt_ip"]); # I think we need to modify to differ between sibling or successor
      ca_act.append       (candidate_train_return["act"]);
      c_nhid_units.append (candidate_train_return["units"]);
      
    elif (candidate_train_return["type"] == "sibling"):
      if (self._params.get_param ("verbose") == True):
        print ("Adding: SIBLING");
      
      # hstack because the input to hidden weights will have columns as hidden node connections.
      ca_W[len (ca_W)-1] = np.hstack ([ca_W[len (ca_W)-1], candidate_train_return["wt_ip"]]);
      c_nhid_units[len (c_nhid_units) - 1] += candidate_train_return["units"];
      
    else:
      print ("ERROR: Invalid unit type: \"", candidate_train_return["type"], "\"");
      # TODO: Stop training here
      pass;
    
    # Record the numbmer of unit type and number
    self._nn_arch_obj.set_this_layer_type ({"hid_type": candidate_train_return["type"], "hid_nos": candidate_train_return["units"]});
    
    wt_candidate_op = wtfunc_p (1, op_W.shape[1]);
    if (ccnn_method == "cascade2" or ccnn_method == "cascade3"):
      wt_candidate_op = candidate_train_return["wt_op"];
      wt_candidate_op = wt_candidate_op[0:(wt_candidate_op.shape[0]-1),:]; # NOTE: The bias weights are already zero as we are setting the bias value as 0
      # This appends random weights. for TESTING only
      #wt_candidate_op = wtfunc_p (wt_candidate_op.shape[0], wt_candidate_op.shape[1]);
    
    # FIXME: TESTING: Here the op_W last row corresponds to the bias unit, therefore we should handle that.
    op_W = np.vstack([op_W[0:(op_W.shape[0]-1),:], wt_candidate_op, op_W[op_W.shape[0]-1,:]]); # Extend now to allow forwand pass
    #print (op_W[0:(op_W.shape[0]-1),:].shape)
    #print (op_W[op_W.shape[0]-1,:].shape)
    self._nn_arch_obj.set_op_layer_weight              (op_W);
    self._nn_arch_obj.set_per_layer_cascade_weight     (ca_W);
    self._nn_arch_obj.set_per_layer_cascade_activation (ca_act);
    self._nn_arch_obj.set_cascade_nhid_units           (c_nhid_units);
    
    # WARNING: Here we need to check if these outputs are repeatedly thresholded or not.
    self._nn_arch_obj.do_forward_pass (train_data["X"]);
    op_h = self._nn_arch_obj.get_op_hidden_layer ();
    op_f = self._nn_arch_obj.get_op_final ();
    
    
    if (self.__validation_flag_p is True):
      self._nn_arch_obj.do_forward_pass (self.__X_valid);
      self.__op_h_valid = self._nn_arch_obj.get_op_hidden_layer ().copy ();
      self.__op_f_valid = self._nn_arch_obj.get_op_final ().copy ();
      self.__op_f_unthresh_valid = self._nn_arch_obj.get_op_unthresh_final().copy ();
    
    
    op_W         = self._nn_arch_obj.get_op_layer_weight ();
    ca_W         = self._nn_arch_obj.get_per_layer_cascade_weight ();
    ca_act       = self._nn_arch_obj.get_per_layer_cascade_activation ();
    c_nhid_units = self._nn_arch_obj.get_cascade_nhid_units ();
    
    # Add candidate layer in network including the activation function
    
    # Train output layer only.
    if (self._params.get_param ("verbose") == True):
      print ("Training Output Layer");
    op_W = self._train_output_layer (np.hstack (op_h), train_data["y"], op_W, self._cost);
    self._nn_arch_obj.set_op_layer_weight (op_W);
    self._nn_arch_obj.do_forward_pass (train_data["X"]);
    op_f = self._nn_arch_obj.get_op_final ().copy ();
    op_unthresh_f = self._nn_arch_obj.get_op_unthresh_final ().copy ();

    ## Compute new output. FIXME: Unnecessary here. Computed by outer loop
    #train_cost_val = self._cost.cost (op_f, train_data["y"]);
    #self.__ccnn_train_errors.append (train_cost_val);
    #if (self._params.get_param ("verbose") == True):
      #print ("New cost ", round (train_cost_val, 6));
      #print ("----------------");
    
    
    # Epochs for which the time is spent. If there are 4 processes, and per process there are 400 epochs, for every layer, then this 
    # will result in [400, 400, 400]. but if there are same numbe of epochs divided in 2 processes, then this will be [800, 800, 800].
    # Therefore this accounts for the number of epochs which actually constituted the computation time in parallel. Although, if the 
    # number of cores is more than the number of schedulable processes, then this will not represent the actual time spent.
    self.real_time_epochs = list (map (lambda z: max (z), list (map (lambda x: list (map (lambda y: sum (y), x)), self.candidate_epochs))));
    
    # END training and installing a layer   
    
    #op_W = self._nn_arch_obj.get_op_layer_weight ();  
    #op_W = self._train_output_layer (np.hstack (op_h), train_data["y"], op_W, self._cost);
    #self._nn_arch_obj.set_op_layer_weight (op_W);
    
    #if (self._params.get_param ("verbose") == True):
      #print ("----------------");
  
  def get_epoch (self):
    return sum (self.real_time_epochs) + self.op_epochs;
  
  def train_batch (self, train_data, batch_idx):
    
    self.ccnn_train (train_data, batch_idx);
    pass;
    
    
  def get_ccnn_train_errors (self):
    return self.__ccnn_train_errors;
    
