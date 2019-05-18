
import numpy as np                                                                          # Library: Matrix Ops
import torch.utils.data as utils                                                            # Library: Pytorch Utilities

np.random.seed(123)

#---------------------------------------
# Class: Custom Pytorch Dataset
# Purpose: For Pytorch Dataloader 
#---------------------------------------

class Custom_Dataset(utils.Dataset):
    
    def __init__(self, dataset):
       
        self.labels = dataset['labels']
        self.samples = dataset['samples']
    
    def __getitem__(self, index):
        
        return self.samples[index], self.labels[index]
    
    def __len__(self):
        
        return len(self.samples)    

#---------------------------------------
# Initialize: Synthetic Data Labels
# Labels: Aggregation Based
#---------------------------------------

def LOSN_labels(data, LOSN):

    if(LOSN.lower() == 'mean'):
        return np.asarray([np.mean(ele) for ele in data])                                   

    elif(LOSN.lower() == 'hardmax'):
        return np.asarray([np.max(ele) for ele in data])

    elif(LOSN.lower() == 'hardmin'):
        return np.asarray([np.min(ele) for ele in data])
    
    else:
        print('\nError: Invalid LOSN Aggregation Type\n')    
        exit()

#---------------------------------------
# Initialize: Synthetic Data Samples
#---------------------------------------

def create_data(samples, features, LOSN_type):

    samples = np.round(np.random.rand(samples, features), 3)                                
    labels = LOSN_labels(samples, LOSN_type)                                                
    return {'samples': samples, 'labels': labels}

#---------------------------------------
# Load: Experiment (Synthetic, XOR)
#---------------------------------------

def load_data(params):
    
    dataset = params['experiment']                                                          # Load: LOSN Experiment
    
    if(dataset.lower() == 'synth'):                                                         # Experiment: Synthetic
        
        choice = params['aggregation']                                                      # Load: Aggregation Type
        valid_samples = params['valid_samples']                                             # Load: Valid Samples
        train_samples = params['train_samples']                                             # Load: Train Samples
        valid_features = params['valid_features']                                           # Load: Valid  Features
        train_features = params['train_features']                                           # Load: Train Features

        train = create_data(train_samples, train_features, choice)                          # Initialize: Train Dataset
        valid = create_data(valid_samples, valid_features, choice)                          # Initialize: Valid Dataset
        
    elif(dataset.lower() == 'xor'):                                                         # Experiment: XOR
    
        data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])                                 # Load: XOR Samples
        labels = np.asarray([0, 1, 1, 0])                                                   # Load: XOR Labels 
        
        train = {'samples': data, 'labels': labels}                                         # Initialize: Train Dataset
        valid = None
    
    else:

        print('\nError, Experiment Options: Synth, XOR\n')
        exit()

    return train, valid 
