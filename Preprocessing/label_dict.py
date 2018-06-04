import glob

def label_dict(path = '/datasets/Caltech256'):
    '''
    To build a dict to record the label
    (need to import glob package)
    Arguments:
    path -- The path of Image Set
    
    Returns:
    dict_l -- return a dict that contains 256 classes
    '''
    all_label = glob.glob(path + '/256_ObjectCategories/*')
    all_label.sort()
    
    dict_l = {}
    class_num = len(all_label) - 1
    for i in range(class_num):
        dict_l[i+1] = all_label[i].split('.')[-1]
    
    return dict_l