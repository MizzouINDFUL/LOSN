
#-----------------------------------------------------------------
# University of Missouri-Columbia
#
# Date: 7/5/2019
# Author: Charlie Veal
# Description: Main Experiment, See Repo For Details 
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

import yaml                                                                                 # Library: Config File
import torch                                                                                # Library: Pytorch 
import datasets                                                                             # Library: Dataset(s)
import argparse                                                                             # Library: System Ops
import numpy as np                                                                          # Library: Matrix Ops
import torch.utils.data as utils                                                            # Library: Pytorch Utils

from plots import *                                                                         # Library: Plotting
from model import train_model                                                               # Library: LOSN 

#---------------------------------------
# Experiment: Initialize Data, LOSN 
#---------------------------------------

def initialize_LOSN(params):
    
    batch = params['batch_size']                                                            # Load: Batch Size
    epochs = params['num_epochs']                                                           # Load: Train Epochs 
    options = params['experiment']                                                          # Load: Experiment 
    visualize_loss = params['visualize_loss']

    train_data, valid_data = datasets.load_data(params)                                     # Load: Experiment Dataset(s)
    
    choice = options['type']                                                                # Load: Discrete || Aggregation || XOR 
    if(choice.lower() == 'dis'):                                                            # Experiment: Classification
         
        unique_labels = np.unique(train_data['labels'])
        num_classes_train = len(unique_labels)

        print('#------------------------------')
        print('# LOSN Training: One vs All')
        print('#------------------------------\n') 
        
        all_results = {}

        results = []
        for target in unique_labels:
        
            train_data['labels'] = np.where(train_data['labels'] == target, 1, 0)           # Update: Labels 1 v All (Train)
            
            if(valid_data is not None):
                valid_data['labels'] = np.where(valid_data['labels'] == target, 1, 0)       # Update: Labels 1 v All (Valid) 
 
            #---------------------------
            # Initialize: DataLoader(s) 
            #---------------------------

            train = datasets.Custom_Dataset(train_data)                                     # Update: Dataset(s) Pytorch
            train = utils.DataLoader(dataset = train, shuffle = True, batch_size = batch)   # Utilize: Pytorch Dataloader (Mini-Batch)
          
            if(valid_data is not None):
                valid = datasets.Custom_Dataset(valid_data)
                valid = utils.DataLoader(dataset = valid, batch_size = batch)  
            else:
                valid = None
 
            params['train'] = train
            params['valid'] = valid    
            params['choice'] = choice
            params['features'] = train_data['features']
            params['progress'] = '1vA '+str(int(target)+1)+'/'+str(len(unique_labels)) 
            
            results.append(train_model(params))                                             # Process: Train Model 

        all_results['results'] = results        

    elif(choice.lower() == 'agg' or choice.lower() == 'xor'):                               # Experiment: Aggregation
        
        if(choice.lower() == 'xor'):
            
            print('#------------------------------')
            print('# LOSN Training: XOR Problem')
            print('#------------------------------\n') 
 
            params['progress'] = 'XOR'

        else:
    
            print('#------------------------------')
            print('# LOSN Training: Aggreation')
            print('#------------------------------\n') 
        
            params['progress'] = 'Aggregation' 
 
        #---------------------------
        # Initialize: DataLoader(s) 
        #---------------------------

        train = datasets.Custom_Dataset(train_data)                                         # Update: Dataset(s) Pytorch
        train = utils.DataLoader(dataset = train, shuffle = True, batch_size = batch)       # Utilize: Pytorch Dataloader (Mini-Batch)
        
        if(valid_data is not None):
            valid = datasets.Custom_Dataset(valid_data)
            valid = utils.DataLoader(dataset = valid, batch_size = batch)  
        else:
            valid = None
 
        params['train'] = train
        params['valid'] = valid    
        params['choice'] = choice
        params['features'] = train_data['features']

        all_results = train_model(params)                                                   # Process: Train Model 
    
    #-----------------------------------
    # Visualization: Plotting Loss  
    #-----------------------------------
     
    if(visualize_loss): 
        all_results['valid'] = valid_data
        all_results['choice'] = choice
        all_results['epochs'] = epochs
        display_loss(all_results)
    
    #-----------------------------------
    # Visualization: Plotting Boundary  
    #-----------------------------------
    
    if(choice.lower() =='xor'): 
        evaluate_model(all_results)
  
#---------------------------------------
# Main: Load Yaml Configuration File
#---------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help = "LOSN Configuration File")
    args = parser.parse_args()
                    
    if(args.config == None):
        print('\nAttach Configuration File! Run experiment.py -h\n')
        exit()
    else:
        print('\nAttempt: Loading ', args.config,'...\n')

    params = yaml.load(open(args.config), Loader=yaml.FullLoader)

    initialize_LOSN(params)

