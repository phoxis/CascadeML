
class threshold_lmfit_c:
  
  def __init__ (self, idealLabels = None, modelOutLabels = None):
    self.__weights = None;
    self.__thresholds = None;
    
    if ((idealLabels is not None) and (modelOutLabels is not None)):
      self.train_threshold (idealLabels, modelOutLabels);
    
  # TODO: Can do crossvalidation for the thresholding
  def train_threshold (self, idealLabels, modelOutLabels):
    numExamples = idealLabels.shape[0];
    numLabels = idealLabels.shape[1];
    
    float_max = np.finfo (float).max
    float_min = np.finfo (float).min

    thresholds = np.array ([0] * numExamples);

    
    for example in range (numExamples):
        isLabelModelOuts = np.array ([float_max] * numExamples);
        isNotLabelModelOuts = np.array ([float_min] * numExamples);
        for label in range (numLabels):
            if (idealLabels[example,label] == 1):
              isLabelModelOuts[label] = modelOutLabels[example,label];
            else:
              isNotLabelModelOuts[label] = modelOutLabels[example,label];
              
        isLabelMin = isLabelModelOuts[np.argmin(isLabelModelOuts)];
        isNotLabelMax = isNotLabelModelOuts[np.argmax(isNotLabelModelOuts)];

        if (isLabelMin != isNotLabelMax):
          if (isLabelMin == float_max):
            thresholds[example] = isNotLabelMax + 0.1;
          elif (isNotLabelMax == float_min):
            thresholds[example] = isLabelMin - 0.1;
          else:
            thresholds[example] = (isLabelMin + isNotLabelMax) / 2;
        else:
            thresholds[example] = isLabelMin;

    modelMatrix = modelOutLabels.copy ();
    modelMatrix = np.insert (modelMatrix, modelMatrix.shape[1], np.array ([1]), axis = 1);
    self.__weights = np.array (np.linalg.lstsq (modelMatrix, thresholds)[0].copy ());
    #return self.__weights[0].copy ()[0:(len (weights[0]-1)];
    
    
  def predict_threshold (self, labelsConfidences):

    expectedDim = self.__weights.shape[0] - 1;
    
    threshold = 0;
    for index in range (expectedDim):
      threshold += labelsConfidences[index] * self.__weights[index];
      
    threshold += self.__weights[expectedDim];

    return threshold;

  def predict_labels (self, labelsConfidences):
    
    labelBipartition = np.zeros_like (labelsConfidences);
    
    for i in range (labelsConfidences.shape[0]):
      threshold = self.predict_threshold (labelsConfidences[i,:]);
      labelBipartition[i,labelsConfidences[i,:]>threshold] = 1;
    
    return labelBipartition.astype (np.bool);

    
  
