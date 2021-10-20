# Speech To Text

Links that I found usefull when working on STT.

## Datasets

* Blog of a list of Voice Datasets -> [9 voice datasets](https://www.cmswire.com/digital-asset-management/9-voice-datasets-you-should-know-about/)
* openSLR dataset -> [LibriSpeech ASR corpus](http://www.openslr.org/12/)
* [VoxForge Dataset](http://www.openslr.org/12/)
* [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
* [Common Voice corpus](https://voice.mozilla.org/en/datasets)
* [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

## Preprocessing Data

* Great blog about audio signal preprocessing -> [Speech Processing](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
* Python Library -> [pyfilterbank](http://siggigue.github.io/pyfilterbank/), [Librosa](https://librosa.github.io/librosa/), [SoundFile](https://github.com/bastibe/SoundFile/)
* Kaggle example -> [Speech Recognition Challenge](https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example)

## Papers

* [Towards End-to-End Speech Recognition with Reccurent Neural Networks](http://proceedings.mlr.press/v32/graves14.pdf)
* [Deep Speech](https://arxiv.org/pdf/1412.5567.pdf)
* [Deep Speech 2](https://arxiv.org/pdf/1512.02595.pdf)
* [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf) (LAS)
* [LAS 2](https://arxiv.org/pdf/1712.01769.pdf)
* [CTC](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.6306&rep=rep1&type=pdf)
* [Syllable-based Seq2Seq Speech Recognition with Transformers](https://arxiv.org/pdf/1804.10752.pdf)
* [Multilingual End-to-End Speech Recognition with A Single Transformer on Low-Resource Languages](https://arxiv.org/pdf/1806.05059.pdf) 

---

My github repository with some ASR experiments -> [ASR](https://github.com/thbeucher/ML_pytorch/tree/master/apop/ASR)

---

Usual approaches for Automatic Speech Recognition (ASR) use different modules (Acoustic Modeling, Pronunciation Modeling, Language Modeling) trained separately and often hand-designed. 

Seeing deep learning improvement (optimization, data usage) over past years, it is possible to consider and design an End-to-End algorithm to perform ASR. 

Multiple architectures were proposed using Recurrent Neural Network (RNN). Two main way appear, one using Connectionist Temporal Classification (CTC) based models and the other using Sequence-to-Sequence framework with Attention. 

All those approaches use Spectrograms as input features and produce character level outputs. They use Beam search and potential Language Model (LM) to improve scores. 

One major problem for ASR with labeled data is the alignment between the audio sources and the transcripts which is partially handle by CTC based models or by the attention mechanism in Seq-to-Seq framework.  

HMM and CTC models made a strong assumption about frame-independence, Seq2Seq models remove this assumption allowing them to learn an implicit LM and optimize WER more directly. (WER - Word Error Rate is the usual metric used to measure ASR models performance) 

All those approaches are computationally heavy, and the recurrence used made parallelization impracticable.

---

1) Input representation

In the litterature, we usually found the use of Filter Banks or Mel-Frequency Cepstral Coefficients (MFCCs) as input features. They comes from extensive human engineering that aim to simulate how the human ear works.

For my Speech-to-Text(STT) project, I've decided to go on an End-to-End design, even for the input. Learning to perform STT task directly from raw signal is really hard and in the current state of our knowledge on artificial neural network(ANN), it requires too much labeled data. When you think about ourself, even with our advance big neural network designed through evolution, we are exposed to a lot of sound signal and it takes us almost a year to show sign that we understand some simple interaction and 3 years and more to be able to use words. We have leverage some labeled data when our parents repeat the same sound when pointing to a single object but most of our learning seems to be self-supervised.
There is a lot of unlabelled sound data in the web, so maybe we can leverage it to allow our ANN to perform well at the task.

The first technic that came to my mind is Predictive Coding(PC) where the theory is that the brain creates and maintains a model of the environment by predicting the futur and compare its prediction to the reality. So, to creates our features representation in a self-supervised way, we can design an ANN that will first downsample the signal, using a convolutional network for example, to obtain a more compact representation (we can add a sparsity constraint on this representation), then we use another neural network, which can also be a convolutional network (using transposed convolution), to recreates the signal and the loss could be a simple Mean Squared Error(MSE). (One work that I found particularly interesting in this field is [SDPC](https://arxiv.org/pdf/1902.07651.pdf))
The problem is that it will be computationally intensive and you will certainly struggle to fit, in your GPU, a big enough architecture that will have a good performance.

Indeed, the usual sample rate of an audio signal is 44.1Khz. It means that you have 44,100 numerical values to describe each second of your record so even with a 2s audio record you will have an array of shape (88200,). In the case of openSLR dataset, the sample rate choosed is way lower, 16Khz, but it still give you big arrays for even small records.

Another option could be a technic called Contrastive Learning (or Contrastive Predictive Coding).

2) Architecture review

Browsing the literature on Speech-to-Text task, we found various kind of architecture. Let's see some and their pytorch implementation.

The most famous one for the moment in NLP field, the Transformer.

![transformer](images/transformer_encoder.png)

Even if there is now an official implementation in pytorch, I've made at the time my own and I've add option to use the version propose in the paper [Stabilizing Transformers For Reinforcement Learning](https://arxiv.org/pdf/1910.06764.pdf). You can find it [here](https://github.com/thbeucher/ML_pytorch/blob/master/apop/models/transformer/encoder.py).

Taking ideas from papers [han2019](https://arxiv.org/pdf/1910.00716.pdf), [wu2019pay](https://openreview.net/pdf?id=SkVhlh09tX), [kriman2020quartznet](https://arxiv.org/pdf/1910.10261.pdf) and [hannun2019sequence](https://arxiv.org/pdf/1904.02619.pdf) we can creates 4 differents convolution architecture:

* Separable Convolution:


* Basic convolutional block:


* Convolution attention from [Lightconv](https://openreview.net/pdf?id=SkVhlh09tX):


* Combination of a ConvBlock and an AttentionConvBlock:


We can also add a simple feed-forward network:


Now with these building blocks, we can create our network:

![arch](images/STT_arch_config.png)

where the features extractor is [wav2vec](https://arxiv.org/abs/1904.05862), the input projection and final projection are simple linears.

The model is made of multiple layers with multiple blocks, to allow multiple configurations using the same class model, we can design our class to be block/layer agnostic.

3) Possible losses

* ASG & CTC -> [ASG-CTC](https://towardsdatascience.com/better-faster-speech-recognition-with-wav2letters-auto-segmentation-criterion-765efd55449), [CTC](https://distill.pub/2017/ctc/), [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
* Cross-Entropy in the case where alignment is handle by using attention mechanism

4) Experiments

You can find the code of an experiment that use the previous described architecture with CTC-loss and obtain a WER of 0.057 in my repository:
[ctc_experiments.py](https://github.com/thbeucher/ML_pytorch/blob/master/apop/ASR/ctc_experiments.py)

---
Site Map:
* Home Page -> *[Home Page](index.md)*
* Transformer -> *[Transformer](transformer.md)*
* Neural Plasticity -> *[Neural Plasticity](plasticity.md)*
* Automatic Speech Recognition -> *[ASR](asr.md)*
* Spiking Neural Network -> *[SNN](snn.md)*
* Curriculum Vitae -> [CV](cv.md)
* Contact page -> *[Contact](contact.md)*
