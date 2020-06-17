# The Transformer

The Transformer was presented in paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the interesting part of the proposed architecture lie in the used attention mechanism that I try to explain in the following post.

You can find an implementation of the Transformer [here](https://github.com/thbeucher/ML_pytorch/blob/master/apop/models/transformer).

## Scaled Dot-Product Attention

The attention mechanism used is depicted by the following equation:
![Scaled Dot-Product Attention Equation](images/scaled_dot_product_attention.png)

where Q, K, V and d<sub>k</sub> are respectively the queries, the keys, the values and the keys dimension.

We can describe what Q, K, V are like:
* Q - query : Input token we currently looking at (a vector)
* K - keys : The tokens we compares to the query (a sequence of vectors)
* V - values : A sequence of vectors used to store the final embedding representation of tokens

**The purpose of this attention mechanism is to create, for each token in a sequence, a new contextual representation that enrich token embedding with information of the other tokens in the sequence.** (it's completely performed in the Transformer architecture by the use of multiple heads and the Feed-Forward network that create our final embedding)

This attention is composed of 3 important steps, let's see them through an example.

We have the following sentence : `my random sentence`

We embeds it into a sequence of vectors : S = [e<sub>my</sub>, e<sub>random</sub>, e<sub>sentence</sub>]

We creates Q, K, V matrices by linearly project each word embedding into embedding of dimension d<sub>k</sub> for the queries and keys, and dimension d<sub>v</sub> for the values (for simplicity, we use d<sub>k</sub>=d<sub>v</sub>=2).

So we have 3 matrices for queries, keys and values projections : W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>

* SW<sub>Q</sub> = Q

![Q](images/Q.png)

* SW<sub>K</sub> = K

![K](images/K.png)

* SW<sub>V</sub> = V

![V](images/V.png)

1) First step - Compares each query to each key

![QK](images/QK.png)

2) Second step - Computes a Score matrix by dividing the matrice obtain at step 1 by sqrt(d<sub>k</sub>) then normalize it using the softmax function

![S](images/S.png)

3) Last step - Computes our final matrice as the weighted sum of word representation with their attention scores

![Vw1](images/Vw1.png)


## Multi-Head Attention

In previous section we depicted the mechanism of one attention head so the multi-head attention is realized by using as many heads as you want then you just have to concatenate the results of each head to finally perform a final projection to obtains the output matrice.

---
Site Map:
* Main Page -> *[Main](index.md)*
* Neural Plasticity post -> *[Neural Plasticity](plasticity.md)*
* Automatic Speech Recognition -> *[ASR](asr.md)*
* Curriculum Vitae -> [CV](cv.md)
* Contact page -> *[Contact](contact.md)*
