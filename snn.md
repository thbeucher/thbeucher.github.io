# Spiking Neural Network and Image Classification
In this blog post we will see in detail (language information, visual information, python code information)
how to implement in python a Spiking Neural Network (SNN) that perform image classification using unsupervised learning algorithm
(Spike-Timing Dependent Plasticity) and supervised one (R-STDP).

## Introduction
What SNN is? STDP? R-STDP?

Task choose = Image Classification -> dataset = MNIST

Overview schema : 

![overview](images/snn_overview_schema.png)

Content:

1. Image transformation using a simple retinal model
2. Temporal transformation using rand-order coding scheme
3. Spiking Deep Convolutional Network Learning using STDP and R-STDP
    1. Neural architecture / Neuron model
    2. Neuron competition / Winners election
    3. STDP and R-STDP learning
4. Hyperparameters
5. Packing everything / Training & Testing script
6. Visualization