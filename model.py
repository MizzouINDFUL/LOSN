
#-----------------------------------------------------------------
# University of Missouri-Columbia
#
# Date: 7/5/2019
# Author: Charlie Veal
# Description: LOSN Model, See Repo For Details 
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

import torch                                                                                # Library: Pytorch
from tqdm import tqdm                                                                       # Library: Progress Bar

torch.manual_seed(123)                                                                      # Code: Pytorch Seed

#---------------------------------------
# Class: Linear Order Statistics Neuron
# Inheritence: Neural Network Module
# Backward Pass: Auto Differentiation
#---------------------------------------

class LOSN(torch.nn.Module):

    def __init__(self, features, bias):
        
        super(LOSN, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.Tensor(features, 1))                        # Initialize: Weights 
        self.weights.data.normal_(-0.1, 0.1)                                                # Populate: Weights (Random Gaussian)
        self.bias_flag = bias

        if(self.bias_flag):                                                                 # Check: Flag, Bias
            self.bias = torch.nn.Parameter(torch.Tensor(1))                                 # Initialize: Bias 
            self.bias.data.normal_(-0.1, 0.1)                                               # Populate: Bias (Random Gaussian)
        
    def forward(self, data):                                                                # Forward Pass
        
        data, data_idx = torch.sort(input=data, descending=True)                            # Process: Non-Linear Sort
        
        if(self.bias_flag):
            prediction = torch.matmul(data, self.weights) + self.bias                       # Process: Dot Product (Bias & Mini-Batch)
        else:
            prediction = torch.matmul(data, self.weights)                                   # Process: Dot Product (Mini-Batch)
        
        return prediction        

#---------------------------------------
# Experiment: Forward / Backward Pass 
#---------------------------------------
   
def train_valid_epoch(params):
        
    data = params['data']                                                                   # Load: Train/Valid Dataset
    cost = params['cost']                                                                   # Load: Cost Function
    model = params['model']                                                                 # Load: Current Model
    optimizer = params['optimizer']                                                         # Load: Model Optimization
    
    epoch_loss = []
    for samples, labels in data:                                                            # Loop: Train/Valid
        
        labels = labels.type('torch.FloatTensor')
        samples = samples.type('torch.FloatTensor')
        
        prediction = model(samples)                                                         # Calculate: Forward Pass
        loss = cost(prediction, labels)                                                     # Calculate: Loss
        epoch_loss.append(loss.item())
    
        if(optimizer is not None):                                                          # Check: Train/Valid
            optimizer.zero_grad()                                                           # Clear: Previous Gradient
            loss.backward()                                                                 # Calculate: Backwards Pass
            optimizer.step()                                                                # Update: Model Parameter(s)
   
    return model, epoch_loss                                                                # Return: Updated Model, Loss
    
#---------------------------------------
# Experiment: Train / Validate Model 
#---------------------------------------

def train_model(params):
   
    bias = params['bias']                                                                   # Load: Flag Bias 
    train = params['train']                                                                 # Load: Train Dataset
    valid = params['valid']                                                                 # Load: Valid Dataset
    verbose = params['verbose']                                                             # Load: Verbose Flag
    marker = params['progress']                                                             # Load: Progress, 1vA
    features = params['features']                                                           # Load: Number Features
    num_epochs = params['num_epochs']                                                       # Load: Number Epochs 
    experiment = params['experiment']                                                       # Load: Experiment Choice
    learning_rate = params['learning_rate']                                                 # Load: Learning Rate
    momentum_rate = params['momentum_rate']                                                 # Load: Momentum Rate
    visualize_loss = params['visualize_loss']                                               # Load: Flag, Loss Plots
 
    #----------------------------------- 
    # Initialize: Model, Parameter(s)
    #----------------------------------- 

    model = LOSN(features, bias)                                                            # Initialize: Model (LOSN)
    cost = torch.nn.MSELoss()                                                               # Initialize: Cost Function 
    optimizer = torch.optim.SGD( model.parameters(), lr = learning_rate, 
                                 momentum = momentum_rate )                                 # Initialize: Optimizer (SGD) 
    
    train_loss, valid_loss = [], []
    for epoch in tqdm(range(num_epochs), desc=marker):                                      # Loop: Train Epochs
        
        if(verbose):
            print('\nTrain: Epoch ', epoch,'\n')
         
        train_params = { 'data': train, 'model': model, 
                         'cost': cost, 'optimizer': optimizer }

        model, epoch_loss = train_valid_epoch(train_params)                                 # Epoch: Training
        train_loss.append(sum(epoch_loss)/len(epoch_loss))                                  # Update: Training Epoch Log 
        
        if(valid is not None):
            
            if(verbose):
                print('\nValid: Epoch ', epoch,'\n')
            
            valid_params = { 'data': valid, 'model': model, 
                             'cost': cost, 'optimizer': optimizer }

            model, epoch_loss = train_valid_epoch(valid_params)                             # Epoch: Validation
            valid_loss.append(sum(epoch_loss)/len(epoch_loss))                              # Update: Training Epoch Log 
    
        if(verbose): 
    
            print('\n#-------------------------------------')
            print('# Epoch Results: ', epoch + 1)
            print('#-------------------------------------\n')

            if(valid is not None):
                
                print( 'Epoch: ', epoch + 1,'\n'
                       'Train-Loss Avergae: ', train_loss[epoch],
                       '\nValid-Loss Average: ', valid_loss[epoch],'\n' )
            else:

                print( 'Epoch: ', epoch + 1,'\n'
                       'Train-Loss Avergae: ', train_loss[epoch], '\n' )
 
    print('\nFinal Train Loss: ', train_loss[-1])
    
    if(valid is not None):
        print('Final Valid Loss: ', valid_loss[-1], '\n')
    else:
        print('\n')
 
    if(params['experiment']['type'].lower() == 'xor'):
        return {'model': model, 'train_loss': train_loss, 'train': train}
    else:
        return {'train_loss': train_loss, 'valid_loss': valid_loss}
