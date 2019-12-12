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
Site Map:
* Main page -> *[Main](index.md)*
* Transformer post -> *[Transformer](transformer.md)*
* Neural Plasticity post -> *[Neural Plasticity](plasticity.md)*
* Curriculum Vitae -> [CV](cv.md)
* Contact page -> *[Contact](contact.md)*
