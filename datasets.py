
#-----------------------------------------------------------------
# University of Missouri-Columbia
#
# Date: 7/5/2019
# Author: Charlie Veal
# Description: Generate Datasets, See Repo For Details 
#-----------------------------------------------------------------
#
# This program is free software: 
# you can redistribute it and / or modify it under the  
# terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be 
# useful, but WITHOUT ANY WARRANTY; without even the implied i
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the 
# GNU General Public License along with this program.  
# If not, see <https://www.gnu.org/licenses/>.
#-----------------------------------------------------------------

import os                                                                                   # Library: OS Ops
import subprocess                                                                           # Library: System Ops
import numpy as np                                                                          # Library: Matrix Ops
import torch.utils.data as utils                                                            # Library: Pytorch Utils

from tqdm import tqdm                                                                       # Library: Progress Bar

np.random.seed(123)                                                                         # Initialize: Numpy Seed 

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
# Labels: Aggregation
#---------------------------------------

def agg_labels(data, choice):

    if(choice.lower() == 'mean'):
        return np.asarray([np.mean(ele) for ele in data])                                   

    elif(choice.lower() == 'hardmax'):
        return np.asarray([np.max(ele) for ele in data])

    elif(choice.lower() == 'hardmin'):
        return np.asarray([np.min(ele) for ele in data])
    
    else:
        print('\nError: Invalid LOSN Aggregation Type\n')    
        exit()

#---------------------------------
# Create: Synthetic Dataset(s)
#---------------------------------

def gen_data(params):
    
    valid = params['use_valid']                                                             # Load: Flag, Valid
    option = params['experiment']                                                           # Load: Experiment
    root_path = params['path_root']                                                         # Load: Root Folder
    filename = params['train_filename']                                                     # Load: Required Filename
    num_samples = params['train_samples']                                                   # Load: Required Samples (Train)
    num_features = params['train_features']                                                 # Load: Required Features (Train)

    if(os.path.exists(root_path) == False):
        subprocess.call(['mkdir','-p', root_path])

    print('#------------------------------')
    print('# Initialize: Data Generation')
    print('#------------------------------\n') 

    datasets = []
    description = 'Gen Train'
    datasets.append([description, filename, num_samples, num_features])    
 
    if(valid):                                                                              # Option: Valid Dataset
        description = 'Gen Valid'
        filename = params['valid_filename']                                                 # Load: Required Filename
        num_samples = params['valid_samples']                                               # Load: Required Samples (Valid)
        num_features = params['valid_features']                                             # Load: Required Features (Valid)
        datasets.append([description, filename, num_samples, num_features])        
    
    labels_choice = option['type']                                                          # Load: Aggregation || Discrete
   
    for current_dataset in datasets:                                                        # Loop: Creation, Train/Valid
        description, filename, num_samples, num_features = current_dataset    
 
        if(labels_choice.lower() == 'agg'):                                                 # Option: Aggregation
            aggregation = option['agg_value']
            samples = np.random.rand(num_samples, num_features)
            labels = agg_labels(samples, aggregation)
            labels = labels.reshape(len(labels), 1) 

        elif(labels_choice.lower() == 'dis'):                                               # Option: Discrete
            num_classes = option['dis_value']
            labels = np.zeros([num_classes * num_samples, 1])
            samples = np.zeros([num_classes * num_samples, num_features])
        
            for current_class in tqdm(range(num_classes), desc=description):                # Loop: Number Classes

                start_pos = num_samples * current_class
                end_pos = start_pos + num_samples
                
                labels[start_pos:end_pos] = current_class
                samples[start_pos:end_pos, :] = np.random.rand(num_samples, num_features)   # Update: Samples
            
        data = np.hstack([samples, labels])                                                 # Merge: Samples, Labels
        
        data_file = open(os.path.join(root_path, filename), 'w')                            # Initialize: Data-File
        for current_sample in tqdm(data, desc='Writing'):                                   # Loop: Write Data --> File
            for count, value in enumerate(current_sample):
                if(count == len(current_sample) - 1):
                    data_file.write(str(value)+'\n')
                else:
                    data_file.write(str(value)+',') 

        print('\nWriting File -- Complete\n')

#---------------------------------------
# Load: Train/Valid Dataset File(s)
#---------------------------------------

def read_dataset(root, filename):

    data = []
    filepath = os.path.join(root, filename)                                                 # Gather: Path, Dataset

    for sample in open(filepath, 'r'):                                                      # Read: Dataset File
         data.append([float(ele) for ele in sample.split(',')])                              
    data = np.asarray(data)        

    samples, labels = data[:, :-1], data[:, -1:]                                            # Gather: Samples, Lables
    features = data[:, :-1].shape[-1]                                                       # Gather: Number Features

    return {'samples': samples, 'labels': labels, 'features': features}                     # Return: Dataset 
    
#---------------------------------------
# Load: LOSN Experiment (Dis, Agg, Xor)
# Note: Dis = Classification Experiment
# Note: Agg = Aggregation Experiment
# Note: Xor = Classical XOR Problem
#---------------------------------------

def load_data(params):
    
    dataset = params['experiment']['type']                                                  # Load: LOSN Experiment
    
    if(dataset.lower() == 'xor'):                                                           # Option: XOR Problem
    
        data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])                                 # Load: XOR Samples
        labels = np.asarray([0, 1, 1, 0])                                                   # Load: XOR Labels 
        features = data.shape[-1]

        train = {'samples': data, 'labels': labels, 'features': features}                   # Initialize: Train Dataset
        valid = None
    
    elif(dataset.lower() == 'dis' or dataset.lower() == 'agg'):                             # Option: Discrete || Aggregation
        
        if(params['gen_data']):                                                             # Flag: Generate Dataset
            gen_data(params)
        
        root = params['path_root']
        train_filename = params['train_filename']
        train = read_dataset(root, train_filename) 
        
        if(params['use_valid']):
            valid_filename = params['valid_filename']      
            valid = read_dataset(root, valid_filename) 
        else:
            valid = None
            
    else:

        print('Error, Experiment Options: Synth, XOR, MNIST, File', '\n')
        exit()

    return train, valid 
