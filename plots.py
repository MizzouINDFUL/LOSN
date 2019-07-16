
#-----------------------------------------------------------------
# University of Missouri-Columbia
#
# Date: 7/5/2019
# Author: Charlie Veal
# Description: Plot Visualizations, See Repo For Details 
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
import numpy as np                                                                          # Library: Matrix Ops
import matplotlib.pyplot as plt                                                             # Library: Plot Ops

from tqdm import tqdm                                                                       # Library: Progress Bar

#-------------------------------------------
# Plots: Display Loss (Train / Validation)
#-------------------------------------------

def display_loss(params):
   
    valid = params['valid']                                                                 # Load: Flag, Validation 
    choice = params['choice']                                                               # Load: Experiment Choice
    num_epochs = params['epochs']                                                           # Load: Num Train Epochs
    
    if(choice.lower() == 'dis'):                                                            # Experiment: Classifcation
        
        all_results = params['results']                                                     # Load: Train Results
        
        if(valid is not None):                                                              # Flag, Plot Validation 
            
            fig, ax = plt.subplots(2)                                                       # Plots: Train/Valid

            ax[0].set_title('Train Loss: 1 vs All')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Error')         
            ax[0].set_ylim(0, 1)
 
            ax[1].set_title('Valid Loss: 1 vs All')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Error')         
            ax[1].set_ylim(0, 1)

            for count, result in enumerate(all_results):                                    # Loop: Train/Valid Results
                
                train = result['train_loss']
                valid = result['valid_loss']

                ax[0].plot( np.arange(num_epochs), train, 
                            linewidth=3.0, label=str('Class: '+str(count)) )                # Plot: Train Loss
                ax[1].plot( np.arange(num_epochs), valid, 
                            linewidth=3.0, label=str('Class: '+str(count)) )                # Plot: Valid Loss
                
            ax[0].legend()
            ax[1].legend()

        else:                                                                               # Plot: Train Loss Only
            
            for count, result in enumerate(all_results):                                    # Loop: Train Results 
                train = result['train_loss']
                plt.plot( np.arange(num_epochs), train, 
                          linewidth=3.0, label=str('Class: '+str(count)) )                  # Plot: Train Loss
            plt.legend()
            plt.title('Train Loss: 1 vs All')
            plt.xlabel('Epochs')
            plt.ylabel('Error')         
            plt.ylim(0, 1)

        plt.tight_layout()
        plt.show()

    else:                                                                                   # Experiment: Aggregation || XOR 

        train = params['train_loss']                                                        # Load: Train Loss
 
        if(valid is not None):                                                              # Flag: Validation
            valid = params['valid_loss']                                                    # Load: Valid Loss
            plt.plot(np.arange(num_epochs), train, linewidth=3.0, label='Train')            # Plot: Train Loss
            plt.plot(np.arange(num_epochs), valid, linewidth=3.0, label='Valid')            # Plot: Valid Loss
            plt.title('Loss: Epoch Train / Validation ')
        else:                                                                               
            plt.plot(np.arange(num_epochs), train, linewidth=3.0, label='Train')
            plt.title('Loss: Epoch Train ')

        plt.xlabel('Epochs')
        plt.ylabel('Error')         
        plt.legend()
        plt.show()

#-------------------------------------------
# Model: Evaluate LOSN, Plot Results
#-------------------------------------------

def evaluate_model(params, res = 0.03, cm = plt.cm.RdBu):
    
    model = params['model']                                                                 # Load: Trained LOSN
    dataset = params['train']                                                               # Load: XOR Dataloader

    print('Evaluating: Test Dataset (LOSN)\n')

    preds, samples, labels = [], [], []
    for sample, label in dataset:                                                           # Loop: Test Dataset
        
        label = label.type('torch.FloatTensor')
        sample = sample.type('torch.FloatTensor')

        samples.append(sample.numpy())                                                      # Gather: Test Samples
        labels.append(label.numpy())                                                        # Gather: Test Labels
    
    labels = np.asarray(labels).reshape(4, 1)
    samples = np.asarray(samples).reshape(4, 2)

    #-----------------------------------
    # Initialize: Mesh Grid (2D Plane)
    #-----------------------------------
    
    x_min, y_min = np.min(samples, 0) - 0.5                                                 # Find: Min, Test Dataset
    x_max, y_max = np.max(samples, 0) + 0.5                                                 # Find: Max, Test Dataset

    x_range = np.arange(x_min, x_max, res)                                                  # Initialize: X-Values (Mesh Grid)
    y_range = np.arange(y_min, y_max, res)                                                  # Initialize: Y-Values (Mesh Grid)
    xx, yy = np.meshgrid(x_range, y_range)                                                  # Initialize: Mesh Grid

    #-----------------------------------
    # Evaluate: Mesh Grid --> Model
    #-----------------------------------

    test_data = np.zeros([len(xx.ravel()), 2])                                              # Initialize: Test Dataset
    test_data[:, 0] = xx.ravel()                                                            # Populate: X Xalues
    test_data[:, 1] = yy.ravel()                                                            # Populate: Y Values
    
    preds = []
    for count, sample in enumerate(tqdm(test_data)):
        sample = torch.tensor(sample).type('torch.FloatTensor')
        preds.append(model(sample).detach().numpy())                                        # Gather: Model Predictions
    preds = np.asarray(preds) 
        
    #-----------------------------------
    # Plot: Model Decision Boundary
    #-----------------------------------

    results = np.asarray(preds).reshape(xx.shape)
    results = np.round(results)

    print('\nPlotting: Decision Boundary (LOSN)\n')
    
    results = []
    for sample in tqdm(preds):
        if(sample >= 0):
            results.append(sample / np.amax(preds))
        else:
            results.append(sample / np.amin(preds))

    results = np.asarray(results).reshape(xx.shape)
    results = np.round(results)

    plt.title('Plot: LOSN XOR ', fontsize = 14)
    plt.contourf(xx, yy, results, cmap = cm)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    #-----------------------------------
    # Plot: Test Dataset Samples
    #-----------------------------------

    all_colors = ['red', 'blue']

    for color, label in enumerate(np.unique(labels)):

        classes = [ele for count, ele in enumerate(samples) if(labels[count] == label)]
        classes = np.asarray(classes)

        plt.scatter(classes[:, 0], classes[:, 1], c = all_colors[color])

    plt.show()
