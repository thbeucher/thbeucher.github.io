# Spiking Neural Network and Image Classification
In this blog post we will see in detail (language/visual/code informations)
how to implement in python a Spiking Neural Network (SNN) that perform image classification using unsupervised learning algorithm
(Spike-Timing Dependent Plasticity) and supervised one (R-STDP).

## Introduction
What SNN is? STDP? R-STDP?

The proposed experiment correspond to the one described in ([Mozafari et al.](https://ieeexplore.ieee.org/document/8356226/)) and implemented by the author [here](https://github.com/miladmozafari/SpykeTorch/blob/master/MozafariShallow.py).
I will use the code from the [SpykeTorch](https://cnrl.ut.ac.ir/SpykeTorch/doc/index.html) library that I will restructure and modify when needed.

So we will use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to show how we can perform Image Classification using SNN and STDP/R-STDP.

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

## Image Transformation using a simple retinal model
We use Difference of Gaussian ([DoG](https://en.wikipedia.org/wiki/Difference_of_Gaussians)) filtering as our retinal model. It well approximates the center-surround property of the Retinal Ganglion Cells (RGC).

So the detection of positive or negative contrasts done by respectively ON-center and OFF-center RGC will be modelled by two DoG kernels. Our filter will correspond to the application of these kernels to an image using convolution operation.

To implement our filter we will first create a function ```construct_DoG_kernel``` and a class ```DoGKernel``` :
![dog_code_visual](images/dog_code_visual.png)
The plots are created with ```kernel_size=1, sigma1=1, sigma2=2``` and using a line of code like this one :
```python
sns.heatmap(dog, linewidths=.5, annot=True, fmt='.2f', cbar=False, xticklabels=False, yticklabels=False)
```
Now we can create our class ```Filter``` : 
```python
class Filter(object):
  def __init__(self, configuration):
    base_configuration = {'kernel_type': DoGKernel,
                          'kernels_conf': [[3, 3/9, 6/9], [3, 6/9, 3/9],
                                          [7, 7/9, 14/9], [7, 14/9, 7/9],
                                          [13, 13/9, 26/9], [13, 26/9, 13/9]],
                          'padding': 6,
                          'threshold': 50, 'use_threshold': True}
    self.configuration = {**base_configuration, **configuration}
    # creates kernels and pad smaller kernels to fit biggest one
    self.max_kernel_size = max([conf[0] for conf in self.configuration['kernels_conf']])
    
    kernels = []
    for conf in self.configuration['kernels_conf']:
      kernel = self.configuration['kernel_type'](*conf)().unsqueeze(0)
      pad = (self.max_kernel_size - conf[0]) // 2
      kernels.append(torch.nn.functional.pad(kernel, (pad, pad, pad, pad)))
        
    self.kernels = torch.stack(kernels)
  
  def __call__(self, x):  # x = (1, 1, height, width)
    out = torch.nn.functional.conv2d(x, self.kernels, padding=self.configuration['padding'])
    
    if self.configuration['use_threshold']:
      out = torch.where(out < self.configuration['threshold'], torch.zeros(1), out)
    return out
```