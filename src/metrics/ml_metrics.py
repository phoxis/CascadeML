#TODO: Make package
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def ml_f_score (pred, target, average = "macro"):
  
  fscores = list ();
  if (average in ["macro", "micro"]):
    for i in range (pred.shape[1]):
      if ((sum (pred[:,i]) == 0) and (sum (target[:,i]) == 0)):
        fscores.append (1.0);
      else:
        fscores.append (f1_score (pred[:,i], target[:,i], average = average));
  elif (average in ["example"]):
    for i in range (pred.shape[0]):
      if ((sum (pred[i,:]) == 0) and (sum (target[i,:]) == 0)):
        fscores.append (1.0);
      else:
        fscores.append (f1_score (pred[i,:], target[i,:], average = "binary"));
                        
  return np.mean (fscores);


# NOTE: This is the version implemented in MULAN and same as the one we implemented in R
def ml_macro_f_score (pred, target, per_label = False):

  # Divide by zero will happen, but we are handling them appropriately
  np.seterr(all = "ignore");  
  macro_precision = np.sum (np.bitwise_and (target, pred), axis = 0) / np.sum (pred, axis = 0);
  macro_recall    = np.sum (np.bitwise_and (target, pred), axis = 0) / np.sum (target, axis = 0);

  macro_precision[np.isnan (macro_precision)] = 1;
  macro_recall[np.isnan (macro_recall)] = 1;
  maf = macro_precision * macro_recall * 2 / (macro_precision + macro_recall);
  maf[np.isnan (maf)] = 0;
  
  if (per_label is True):
    return maf;
  else:
    return np.mean (maf);
  np.seterr(all = "warn");

def ml_hamming_loss (pred, target):
  
  return np.sum (pred != target) / (target.shape[0] * target.shape[1]);

  
def ml_roc_auc (pred, target):
  pred = (pred + 1) / 2;
  target = (target + 1) / 2;

  rocauc_scores = list ();
  for i in range (pred.shape[1]):
    rocauc_scores.append (roc_auc_score (target[:,i], pred[:,i]));
    
  return rocauc_scores;

