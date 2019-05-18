
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

#-------------------------------------------
# Plots: Display Loss (Train / Validation)
#-------------------------------------------

def display_loss(num_epochs, train_loss, valid_loss, valid):

    if(valid is not None):
        plt.plot(np.arange(num_epochs), train_loss, linewidth=3.0, label='Train')
        plt.plot(np.arange(num_epochs), valid_loss, linewidth=3.0, label='Valid')
        plt.title('Loss: Epoch Train / Validation ')
    else:
        plt.plot(np.arange(num_epochs), train_loss, linewidth=3.0, label='Train')
        plt.title('Loss: Epoch Train ')

    plt.xlabel('Epochs')
    plt.ylabel('Error')         
    plt.legend()
    plt.show()

#-------------------------------------------
# Model: Evaluate LOSN, Plot Results
#-------------------------------------------

def evaluate_model(dataset, model, res = 0.03, cm = plt.cm.RdBu):
    
    print('Evaluating: Test Dataset (LOSN)\n')

    preds, samples, labels = [], [], []
    for sample, label in dataset:                                                           # Loop: Test Dataset
        
        label = label.type('torch.FloatTensor')
        sample = sample.type('torch.FloatTensor')

        samples.append(sample.numpy())                                                      # Gather: Test Samples
        labels.append(label.numpy())                                                       # Gather: Test Labels
    
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

    plt.title('Plot: LOSN ', fontsize = 14)
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
