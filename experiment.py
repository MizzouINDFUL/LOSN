
import yaml                                                                                 # Library: Config File
import torch                                                                                # Library: Pytorch 
import plots                                                                                # Library: Custom Plot(s)
import datasets                                                                             # Library: Custom Dataset(s)
import argparse                                                                             # Library: System Ops
import numpy as np                                                                          # Library: Matrix Ops
import torch.utils.data as utils                                                            # Library: Pytorch Utilities

from tqdm import tqdm                                                                       # Library: Progress Bar

torch.manual_seed(123)

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
    for samples, labels in tqdm(data):                                                      # Loop: Train/Valid
        
        labels = labels.type('torch.FloatTensor')
        samples = samples.type('torch.FloatTensor')
        
        prediction = model(samples)                                                         # Calculate: Forward Pass
        loss = cost(prediction, labels)                                                     # Calculate: Loss
        epoch_loss.append(loss.item())
    
        if(optimizer is not None):                                                          # Check: Train/Valid
                
            optimizer.zero_grad()                                                           # Clear: Previous Gradient
            loss.backward()                                                                 # Calculate: Backwards Pass
            optimizer.step()                                                                # Update: Model Parameter(s)
   
    return model, epoch_loss
    
#---------------------------------------
# Experiment: Train / Validate Model 
#---------------------------------------

def train_model(params):
    

    bias = params['bias']                                                                   # Load: Flag Bias 
    train = params['train']                                                                 # Load: Train Dataset
    valid = params['valid']                                                                 # Load: Valid Dataset
    experiment = params['experiment']                                                       # Load: Exp Dataset
    num_epochs = params['num_epochs']                                                       # Load: Number Epochs 
    learning_rate = params['learning_rate']                                                 # Load: Learning Rate
    momentum_rate = params['momentum_rate']                                                 # Load: Momentum Rate
    visualize_loss = params['visualize_loss']                                               # Load: Flag, Loss Plots
    
    if(experiment.lower() == 'xor'):                                                        # Check: Experiment
       features = params['xor_features']                                                    # Load: XOR Features
    else:
       features = params['train_features']                                                  # Load: Synth Features
 
    print('\n#---------------------------') 
    print('# Initialize: Model, Params ')
    print('#---------------------------\n') 
    
    print('Process -- Complete')
 
    model = LOSN(features, bias)                                                            # Initialize: Model (LOSN)
    cost = torch.nn.MSELoss()                                                               # Initialize: Cost Function 
    optimizer = torch.optim.SGD( model.parameters(), lr = learning_rate, 
                                 momentum = momentum_rate )                                 # Initialize: Optimizer (SGD) 
    
    train_loss, valid_loss = [], []
    for epoch in range(num_epochs):                                                         # Loop: Train Epochs
        
        print('\nTrain: Epoch ', epoch,'\n')
         
        train_params = { 'data': train, 'model': model, 
                         'cost': cost, 'optimizer': optimizer }

        model, epoch_loss = train_valid_epoch(train_params)                                 # Epoch: Training
        train_loss.append(sum(epoch_loss)/len(epoch_loss))                                  # Update: Training Epoch Log 
        
        if(valid is not None):
            print('\nValid: Epoch ', epoch,'\n')
            
            valid_params = { 'data': valid, 'model': model, 
                             'cost': cost, 'optimizer': optimizer }

            model, epoch_loss = train_valid_epoch(valid_params)                             # Epoch: Validation
            valid_loss.append(sum(epoch_loss)/len(epoch_loss))                              # Update: Training Epoch Log 
 
        print('\n#-------------------------------------')
        print('# Epoch Results: ', epoch + 1)
        print('#-------------------------------------\n')

        if(valid is not None):
            
            print( 'Epoch: ', epoch + 1,'\n'
                   'Train-Loss Avergae: ', train_loss[epoch],
                   '\nValid-Loss Average: ', valid_loss[epoch] )
        else:

            print( 'Epoch: ', epoch + 1,'\n'
                   'Train-Loss Avergae: ', train_loss[epoch] )
    
    print('\n#-------------------------------------\n')

    #-----------------------------------
    # Visualization: Plotting Loss  
    #-----------------------------------
    
    if(visualize_loss): 
        plots.display_loss(num_epochs, train_loss, valid_loss, valid)
    
    #-----------------------------------
    # Visualization: Plotting Boundary  
    #-----------------------------------
    
    if(experiment.lower() =='xor'): 
        plots.evaluate_model(train, model)

#---------------------------------------
# Experiment: Initialize Dataset(s) 
#---------------------------------------

def initialize_train(params):
    
    batch = params['batch_size']                                                            # Load: Batch Size
    train, valid = datasets.load_data(params)                                               # Load: Experiment Dataset(s)
    
    print('#---------------------------') 
    print('# Initialize: All Datasets')
    print('#---------------------------\n') 
    
    train = datasets.Custom_Dataset(train)                                                  # Update: Dataset(s) Pytorch
    train = utils.DataLoader(dataset = train, shuffle = True, batch_size = batch)           # Utilize: Pytorch Dataloader (Mini-Batch)
  
    if(valid is not None):
        valid = datasets.Custom_Dataset(valid)
        valid = utils.DataLoader(dataset = valid, shuffle = False, batch_size = batch)  
    
    params['train'] = train
    params['valid'] = valid    
    
    print('Process -- Complete')

    train_model(params)                                                                     # Process: Train Model 
    
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
        print('Attempt: Loading ', args.config,'...\n')

    params = yaml.load(open(args.config), Loader=yaml.FullLoader)
    initialize_train(params)
                                                                        
