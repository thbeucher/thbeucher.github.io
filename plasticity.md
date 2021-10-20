Following papers:
* [Learning to learn with backpropagation of Hebbian plasticity](https://arxiv.org/pdf/1609.02228.pdf) [2016]
* [Differentiable plasticity : training plastic neural networks with backpropagation](https://arxiv.org/pdf/1804.02464.pdf) [2018]
* [BackPropamine : training self-modifying neural networks with differentiable neuromodulated plasticity](https://openreview.net/pdf?id=r1lrAiA5Ym) [2019]

![neuron](images/neuron_synapse_dendrites.jpg)

Some definition from [Artificial Evolution of Plastic Neural Network : a Few Key Concepts](https://hal.archives-ouvertes.fr/hal-01300702/document)
and [NonSynaptic Plasticity](https://en.wikipedia.org/wiki/Nonsynaptic_plasticity) :

<code>Hebb's rule</code> : neurons that fire together, wire together. (if a neuron repeatedly takes part
in making another neuron fire, the connection between them is strengthened)

<code>Structural plasticity</code> : the mechanism describing a generation of new connections and thereby redefining
the topology of the network.

<code>Synaptic plasticity</code> : the mechanism of changing strength values of existing connections.

<code>Non-Synaptic Plasticity</code> : modification of intrinsic excitability of the neuron.
Excitatory postSynaptic potentials (EPSPs) and Inhibitory postSynaptic potentials (IPSPs).

About the papers :

* *Fact* : usual Neural Network trained with backpropagation have fixed connections weights that do not change once the training is complete.
* *Argument* : in nature, the primary basis for long-term learning and memory in the brain is *synaptic plasticity*
* *Challenge* : optimizing through gradient descent not only the base weights, but also the amount of plasticity in each connection.
* *Result* : fixed parameters obtained but describing how to change each connection over time.

In [[2016](https://arxiv.org/pdf/1609.02228.pdf)] paper they propose a time-dependent quantity for each connection
in the network, called the **Hebbian trace** :

<code>Hebb<sub>k</sub>(t) = (1 - &gamma;) * Hebb<sub>k</sub>(t-1) + &gamma; * x<sub>k</sub>(t) * y(t)</code>

where <code>y(t)</code> is the activity of the post-synaptic cell, <code>x<sub>k</sub>(t)</code> is the
activity of the pre-synaptic cell, and <code>&gamma;</code> is a time constant.

So the response of a given cell can be written with a fixed component (classic weights) and a plastic one : 

<code>y(t) = tanh(&Sigma;<sub>k</sub> w<sub>k</sub>x<sub>k</sub>(t) + &alpha;<sub>k</sub>Hebb<sub>k</sub>(t)x<sub>k</sub>(t) + b)</code>

here the plastic parameter <code>&alpha;<sub>k</sub></code> is constant over time.

In [[2018](https://arxiv.org/pdf/1804.02464.pdf)] paper, they rebaptised <code>&gamma;</code> as <code>&eta;</code> (call it the learning rate of plasticity) and change the **Hebbian trace** definition using Oja's rule.
Oja's rule is a modification of the standard Hebb's rule that allow, among others, to maintain stable weight values indefinitely in the absence of stimulation (thus allowing stable long-term memories) while still preventing runaway divergences.

The **Hebbian trace** definition become :

<code>Hebb<sub>k</sub>(t+1) = Hebb<sub>k</sub>(t) + &eta;y(t)(x<sub>k</sub>(t-1) - y(t)Hebb<sub>k</sub>(t))</code>

In fact, they change the notation as follow :

<code>Hebb<sub>i,j</sub>(t+1) = Hebb<sub>i,j</sub>(t) + &eta;x<sub>j</sub>(t)(x<sub>i</sub>(t-1) - x<sub>j</sub>(t)Hebb<sub>i,j</sub>(t))</code>

as, in the context of recurrent neural network, they put the plasticity component with the hidden state passed from previous step *t-1* (<code>x<sub>i</sub>(t-1)</code>) and current input gate (<code>x<sub>j</sub>(t)</code>). And instead of speaking about the connection of a neuron *k* with the current looked neuron, they use the notation for representing the connection between neurons *i* and *j*.

The differentiable plasticity framework from [[2016](https://arxiv.org/pdf/1609.02228.pdf)] facilitated the automatically weight changes as a function of pre- and post-synaptic activity. In [[2019](https://openreview.net/pdf?id=r1lrAiA5Ym)] paper, they propose to extend the framework with neuromodulated plasticity (ie allowed the network to actively modulate the connections plasticity).

First the **Hebbian trace** definition become :

<code>Hebb<sub>i,j</sub>(t+1) = Clip(Hebb<sub>i,j</sub>(t) + &eta;x<sub>i</sub>(t-1)x<sub>j</sub>(t))</code>

where the **Clip** function can be :
* a simple decay term (as in [[2016](https://arxiv.org/pdf/1609.02228.pdf)] paper)
* a normalization implementing Oja's rule (as in [[2018](https://arxiv.org/pdf/1804.02464.pdf)] paper)
* a simple hard clip (keeping value in [-1, 1], as proposed in this new paper)

Two types of neuromodulation is proposed :

1) a simple one that replace the time-fixed parameter <code>&eta;</code> by a time-varying scalar <code>M(t)</code> computed by the network. The **Hebbian trace** definition become : 

<code>Hebb<sub>i,j</sub>(t+1) = Clip(Hebb<sub>i,j</sub>(t) + M(t)x<sub>i</sub>(t-1)x<sub>j</sub>(t))</code>

2) a more sophisticated one that try to mimic the effect of dopamine on plasticity. Here, an **Eligibility trace** is defined as a simple exponential average of the Hebbian product of pre- and post-synaptic activity with trainable decay factor <code>&eta;</code>. Moreover the **Hebbian trace** is gated by the current dopamine signal <code>M(t)</code>. That give use the following equations:

<code>Hebb<sub>i,j</sub>(t+1) = Clip(Hebb<sub>i,j</sub>(t) + M(t)E<sub>i,j</sub>(t))</code>

<code>E<sub>i,j</sub>(t+1) = &eta;x<sub>i</sub>(t-1)x<sub>j</sub>(t) + (1 - &eta;)E<sub>i,j</sub>(t)</code>

---
Site Map:
* Home Page -> *[Home Page](index.md)*
* Transformer -> *[Transformer](transformer.md)*
* Neural Plasticity -> *[Neural Plasticity](plasticity.md)*
* Automatic Speech Recognition -> *[ASR](asr.md)*
* Spiking Neural Network -> *[SNN](snn.md)*
* Curriculum Vitae -> [CV](cv.md)
* Contact page -> *[Contact](contact.md)*
