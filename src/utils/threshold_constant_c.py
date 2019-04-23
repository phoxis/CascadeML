class threshold_constant_c:
  
  def __init__ (self, threshold = None):
    self.__threshold = threshold;
    
  def train_threshold (self, threshold):
    self.__threshold = threshold;
    pass;
    
  def predict_threshold (self, labelsConfidences):
    return self.__threshold;

  def predict_labels (self, labelsConfidences):
    
    labelBipartition = np.zeros_like (labelsConfidences);
    
    for i in range (labelsConfidences.shape[0]):
      threshold = self.predict_threshold (labelsConfidences[i,:]);
      labelBipartition[i,labelsConfidences[i,:]>threshold] = 1;
    
    return labelBipartition.astype (np.bool);

