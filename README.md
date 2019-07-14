# Linear Order Statistic Neuron (LOSN)<br>

## Links

For full details, read

C. Veal, A. Yang, A. Hurt, 
M. Islam, D. T. Anderson, G. Scott, T. Havens, J. M. Keller, B. Tang, 
"Linear Order Statistic Neuron," to appear in FUZZ-IEEE, 2019

Pre-print can be accessed at

http://derektanderson.com/pdfs/LOSN.pdf

## What is a LOSN?

A linear order statistic neuron (LOSN) is an aggregation operator/function. 
LOSN is a generalization of the popular ordered weighted average (OWA) operator. 
A LOSN is used for a variety of tasks. 
Examples include: Multi-criteria Decision Making, Regression, Computer Vision, 
Signal/Image processing, etc. 
In our paper, "Linear Order Statistic Neuron", we discuss its role in pattern recognition. 

## Code Overview

This code is broken down into the following files:

1. **config.py** - Configuration File
2. **experiment.py** - Main File. Includes Training
3. **datasets.py** - Loads Dataset: XOR || Synthethic 
4. **plots.py** - Displays Training Loss, Results

This code supports the following functionality:

1. Train LOSN to learn aggregation
2.  Train LOSN to solve XOR problem
