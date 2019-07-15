# Linear Order Statistic Neuron (LOSN)<br>

## Paper Link

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

### Inspiration

<p align="center">
  <img src="https://user-images.githubusercontent.com/46911156/61239562-c8146b80-a704-11e9-995c-e7b8fcfafbc5.png" width="750">
</p>

The perceptron is an iconic pattern recognition tool that serves as a basis for many renown supervised models. The OWA is an iconic fuzzy learning tool that allows one to explore a multitude of different aggregations. However, when examined closely, the mathematics of the perceptron and the OWA are very similar. Herein, the inspriation, of the LOSN, is the question: What if one applies the OWA to pattern recognition?  

### Mathematics

<p align="center">
  <img src="https://user-images.githubusercontent.com/46911156/61193168-60680d00-a67f-11e9-90d1-3d24dd0b0e7d.png" width="256"> </p>

The LOSN can be mathematically expressed as the sum of a bias and dot product of weights and a sorted input. One can can see this is equivalent to the OWA expression above, except with an additional bias term. 

### Visualization

<p align="center">
  <img src="https://user-images.githubusercontent.com/46911156/61239838-73252500-a705-11e9-9efb-cbb5ddd7d657.png" width="512"> </p>

This is a visualization of the LOSN from a pattern recognition perspective. Referencing the LOSN equation above, one will notice the sort on the inputs. This sort on a cartesian plane can be represented by N! different regions of space. Since this is 2D space, then we have 2 regions: Red and Blue. The dotted line between the regions represent the area where they are equal to one another. The LOSN is actually creating N! perceptrons, one for each region, which inserect in space. These N! perceptrons form a non-linear hyper-wedge, which allows the LOSN to approach non-linear problems. 

                                                                                                                           
## Code Overview

### Installation

The recommendation for installtion is anaconda/miniconda: <p><a href="https://www.anaconda.com/distribution/">Anaconda</a> 
<a href="https://docs.conda.io/en/latest/miniconda.html">Miniconda</a></p>

#### Terminal (Linux Distros, MacOS, WSL, Anaconda Prompt)

Always recommended to create isolated conda environment.

Pick Pytorch GPU or Pytorch CPU (not both!)

**General Packages**: conda install -c conda-forge tqdm pyyaml matplotlib

**Pytorch GPU**: conda install -c pytorch pytorch torchvision

**Pytorch CPU**: conda install -c pytorch pytorch-cpu torchvision-cpu 

**Link**: Official pytorch install recommendation: <a href="https://pytorch.org/">Pytorch</a>

### File Description

This code is broken down into the following files:
1. **config.py** - Configuration File. 
2. **experiment.py** - Main File. Includes Training
4. **plots.py** - Displays Training Loss, XOR Visualization
3. **datasets.py** - Loads Dataset: XOR || Synthethic Classification || Synethtic Aggregation
 
### How to Use

This code supports the following functionality:

1. Train LOSN to solve synthetic classification.
2. Train LOSN to learn synthetic aggregation
2. Train LOSN to solve XOR problem



