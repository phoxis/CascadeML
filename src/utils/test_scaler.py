def mulan_scaler (data, scale_range):
  
  data = data.astype (np.double);
  idx_hash = dict ();
  for j in range (data.shape[1]):
    idx_hash[j] = [np.min (data[:,j]), np.max (data[:,j])];
  
  for i in range (data.shape[0]):
    for j in range (data.shape[1]):
      if (idx_hash[j][0] == idx_hash[j][1]):
        data[i,j] = scale_range[0];
        #print (idx_hash[j])
      else:
        #print (idx_hash[i], idx_hash[i][1] == idx_hash[i][0])
        #print (scale_range)
        #print (data[i,j])
        #print ("---");
        data[i,j] = (((data[i,j] - idx_hash[j][0]) / (idx_hash[j][1] - idx_hash[j][0])) * (scale_range[1] - scale_range[0])) + scale_range[0];
        
      #data[i,:] = ((data[i,:] - idx_hash[j][0]) / (idx_hash[j][1] - idx_hash[j][0]) * (scale_range[1] - scale_range[0])) + scale_range[0];

    
  return data;
