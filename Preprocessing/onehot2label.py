import numpy as np
def onehot2label(onehot):
    '''
    This function aim to convert onehot matrix to label vector
    
    Arguments:
    onehot -- (m, num_class)
    num_class -- number of classes
    
    Returns:
    labels -- convert onehot to label
    '''
    
    m = onehot.shape[0]
    labels = np.zeros((m, 1))
    
    for i in range(m):
        labels[i] = int(np.where(onehot[i] == 1)[0][0])
    
    return labels