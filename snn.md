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
2. Temporal transformation using rank-order coding scheme
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
                          'kernels_conf': [[7, 1, 2], [7, 2, 1]],
                          'padding': 3,
                          'threshold': 50, 'use_threshold': True}
    self.configuration = {**base_configuration, **configuration}
    # creates kernels and pad smaller kernels to fit biggest one
    self.max_kernel_size = max([conf[0] for conf in self.configuration['kernels_conf']])
    
    kernels = []
    for conf in self.configuration['kernels_conf']:
      # kernel shape = (1, kernel_size, kernel_size)
      kernel = self.configuration['kernel_type'](*conf)().unsqueeze(0)
      pad = (self.max_kernel_size - conf[0]) // 2
      kernels.append(torch.nn.functional.pad(kernel, (pad, pad, pad, pad)))
    
    # self.kernels shape = (n_kernels, 1, kernel_size, kernel_size) 
    self.kernels = torch.stack(kernels)
  
  def __call__(self, x):  # x = (1, 1, height, width)
    out = torch.nn.functional.conv2d(x, self.kernels, padding=self.configuration['padding'])
    
    if self.configuration['use_threshold']:
      out = torch.where(out < self.configuration['threshold'], torch.zeros(1), out)
    return out
```
As you see, we add an option to allow the DoG cells to output their values only if their activations are above a certain threshold. It will clean the image by removing noise and keep only the most important informations as you can see in the example below on an astronaut image : 

![astronaut](images/astronaut_dog_filtering_w_or_wo_threshold.png)

The above plot is generated on a jupyter notebook using the following code : 

<details>
<summary><strong>Click to expand!</strong></summary>
<p>

```python
%matplotlib inline
import torch
import ipyplot
import torchvision.transforms as transforms

from PIL import Image
from skimage import data

img_arr = data.astronaut()
gray_img_arr = transforms.functional.rgb_to_grayscale(torch.from_numpy(img_arr).permute(2, 0, 1))
gray_img = Image.fromarray(gray_img_arr.squeeze(0).numpy().astype(np.uint8), mode='L')

# Filter is the class implemented above
filters = Filter({'kernels_conf': [[7, 1, 2]], 'padding': 3, 'use_threshold': False})
filters_threshold = Filter({'kernels_conf': [[7, 1, 2]], 'padding': 3})

gray_img_filtered = filters(gray_img_arr.float().unsqueeze(0))
gray_img_filtered_threshold = filters_threshold(gray_img_arr.float().unsqueeze(0))

img_filtered = Image.fromarray(gray_img_filtered.squeeze(0).squeeze(0).numpy().astype(np.uint8), mode='L')
img_filtered_threshold = Image.fromarray(gray_img_filtered_threshold.squeeze(0).squeeze(0).numpy().astype(np.uint8), mode='L')
ipyplot.plot_images([gray_img, img_filtered, img_filtered_threshold], ['original', 'DoG filtered', 'DoG filtered with thresholding'], max_images=3, img_width=300)
```

</p>
</details>

Now that we are able to mimic our retinal ganglion cells and obtain a contrast level information, we can apply the choosen neural coding scheme.

Here a schema that resume and visualize what we have done in this first step : 

![step1_schema](images/img_transfo_using_DoG.png)

## Temporal transformation using rank-order coding scheme
Now we want to convert contrast informations into spikes where the DoG cells, [retinotopically arranged](https://medical-dictionary.thefreedictionary.com/retinotopic+map), that are the most activated (ie highest contrast) will fire first. This simple scheme have demonstrate ([Rullen and Thorpe](https://www.researchgate.net/publication/11952248_Rate_Coding_Versus_Temporal_Order_Coding_What_the_Retinal_Ganglion_Cells_Tell_the_Visual_Cortex)) efficient information transmission and it's computationally very simple.

We will add to our 3-D tensor ```(n_DoG_kernels, image_height, image_width)``` a fourth dimension that correspond to time.
But before that I have to add some context to explain how we will perform our temporal transformation.

In this experiment we use the biological neuron model called [Perfect Integrate-and-fire](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire)
that simply accumulates input spikes from presynaptic neurons and emit a spike if the membrane potential reach a specific threshold.

Moreover, to mimic the visual cortex organization of the brain, we organize the neurons in a retinopically way using convolution operation and shared sets of synaptic weights as each formed map should detect the same visual feature but at different locations.

So a naive implementation could be to process timestep by timestep and keeping in memory the accumulate potential value of every neuron membrane to be able to know when each neuron emit or not a spike. But it's possible to obtain the output train spikes doing only one 2-D convolution computation by create our input spikes in an accumulative manner.

Let's create our ```cumulative_intensity_to_latency``` function that will operate this transformation : 

```python
def cumulative_intensity_to_latency(intensities, n_time_steps, to_spike=True):
  # bin size to get all non-zero intensities into the n_time_steps defined
  bin_size = torch.count_nonzero(intensities) // n_time_steps

  # flattening intensities then sort it
  intensities_flattened_sorted = torch.sort(intensities.view(-1), descending=True)

  # split intensities into n_time_steps bins
  sorted_bins_value = torch.split(intensities_flattened_sorted[0], bin_size)
  sorted_bins_idx = torch.split(intensities_flattened_sorted[1], bin_size)
  
  spike_map = torch.zeros(intensities_flattened_sorted[0].shape)
  
  # create spikes map step by step in a cumulative manner
  # ie when a neuron emite a spike, we propagate it to all next timesteps
  bins_intensities = []
  for i in range(n_time_steps):
    spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])  # cumulative line
    bins_intensities.append(spike_map.clone().reshape(intensities.shape[1:]))
  
  # shape = (n_timesteps, n_filters, height, width)
  out = torch.stack(bins_intensities)
  return out.sign() if to_spike else out
```
Let's try to visualize this transformation using a fake example : 

![cumulative_schema](images/cumulative_intensity_to_latency_schema.png)

And what this transformation on MNIST data look like : 

![intensity_to_latency](images/intensity_to_latency2.png)

Now that we have our inputs as spike trains we can define our network and how the learning is performed.

## Spiking Deep Convolutional Network Learning using STDP and R-STDP

In this experiment we use the architecture from ([Mozafari et Al](https://arxiv.org/abs/1804.00227)) depicted in the following schema : 

![architecture](images/SDCNN_architecture.png)

So we have 3 convolution layers and 3 pooling layers. Only the convolution layers are trainable, they are composed of non-leaky integrate-and-fire neurons retinotopically organized.

Site Map:
* Transformer -> *[Transformer](transformer.md)*
* Neural Plasticity -> *[Neural Plasticity](plasticity.md)*
* Automatic Speech Recognition -> *[ASR](asr.md)*
* Spiking Neural Network -> *[SNN](snn.md)*
* Curriculum Vitae -> [CV](cv.md)
* Contact page -> *[Contact](contact.md)*