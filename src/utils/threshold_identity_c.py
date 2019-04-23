class threshold_identity_c:
  
  def __init__ (self, threshold = None):
    self.__threshold = None;
    
  def train_threshold (self, threshold):
    pass;
    
  def predict_threshold (self, labelsConfidences):
    return self.__threshold;

  def predict_labels (self, labelsConfidences):
    return labelsConfidences;
  

    
  
 
