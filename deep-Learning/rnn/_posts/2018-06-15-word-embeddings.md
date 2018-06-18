---
layout: post
title: "Word embeddings practical questions"
description: "Course RNN - Week 2 - Practical questions"
categories: []
tags: [word-embeddings-tutorial]
redirect_from:
  - /2018/06/15/
---


1. Suppose you learn a word embedding for a vocabulary of 10000 words. Then the embedding vectors should be 10000 dimensional, so as to capture the full range of variation and meaning in those words.
    + True
    + **False**
      + The dimension of word vectors is usually smaller than the size of the vocabulary. Most common sizes for word vectors ranges between 50 and 400.

2. What is [t-SNE (t-Distributed Stochastic Neighbor Embedding)](https://lvdmaaten.github.io/tsne/)?
    + A linear transformation that allows us to solve analogies on word vectors
    + **A non-linear dimensionality reduction technique**
    + A supervised learning algorithm for learning word embeddings
    + An open-source sequence modelin library

3. Suppose you download a pre-trained word embedding which has been trained on a huge corpus of text. You then use this word embedding to train an RNN for a language task of recognizing if someone is happy from a short snippet of text, using a small training set.

    | x (input text)               | y (happy?) |
    |------------------------------|------------|
    | I'm feeling wonderful today! | 1          |
    | I'm bummed my cat is ill.    | 0          |
    | Really enjoying this!        | 1          |

    Then even if the word *"ecstatic"* does not appear in your small training set, your RNN might reasonably be expected to recognize *"I’m ecstatic"* as deserving a label $y=1$.
    + **True**
        + Yes, word vectors empower your model with an incredible ability to generalize. The vector for *"ecstatic"* would contain a positive/happy connotation which will probably make your model classfied the sentence as a *"1"*.
    + False

4. Which of these equations do you think should hold for a good word embedding? (Check all that apply)?
    + [x] $e_{boy} - e_{girl} \approx e_{brother} - e_{sister}$
    + [ ] $e_{boy} - e_{girl} \approx e_{sister} - e_{brother}$
    + [x] $e_{boy} - e_{brother} \approx e_{girl} - e_{sister}$
    + [ ] $e_{boy} - e_{brother} \approx e_{siter} - e_{girl}$

5. Let $E$ be an embedding matrix, and let $O_{1234}$ be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don't we call $E * O_{1234}$ in Python?
    + **It is computationally wasteful.**
    + The correct formula is $E^T * O_{1234}$.
    + This doesn't handle unknown words (`<UNK>`).
    + None of the above: Calling the Python snipped as described above is fine.

6. When learning word embeddings, we create an artificical task of estimating $P(target | context)$. It is okay if we do poorly on this artificial prediction task, the more important by-product of this task is that we learn a useful set of word embeddings.
    + **True**
    + False

7. In the word2vec algorithm, you estimate $P(t | c )$ where $t$ is the target word and $c$ is a context word. How are $t$ and $c$ chosen from the training set? Pick the best answer.
    + $c$ is the one word that comes immediately before $t$.
    + $c$ is the sequence of several words immediately before $t$.
    + **$c$ and $t$ are chosen to be nearby words.**
    + $c$ is the sequence of all the words in the sentence before $t$.

8. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The **word2vec** model uses the following softmax function:
    $$P(t | c) =  \frac{e^{\theta^T_t}e_c}{\sum^{10000}_{t'=1}e^{\theta^{T}e_c}}$$
    + [x] $\theta_t$ and $e_c$ are both 500 dimensional vectors.
    + [ ] $\theta_t$ and $e_c$ are both 1000 dimensional vectors.
    + [x] $\theta_t$ and $e_c$ are both trained with an optimization algorithm such as Adam or gradient descent.
    + [ ] After training, we should expect $\theta_t$ to be very close to $e_c$, when $t$ and $c$ are the same word.

9. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The GloVe model minimizes this objective:
    $$ min\sum^{10000}_{i=1}\sum^{10000}_{j=1}f(X_{ij}(\theta^T_i e_j + b_i + b^t_j - log(X_{ij}))) $$
    Which of these statements are correct? Check all that apply.
    + [ ] $\theta_j$ and $e_j$ should initialized to $0$ at the beginning of training. 
    + [x] $\theta_j$ and $e_j$ should initialized randomly at the beginning of training.
    + [x] $X_{ij}$ is the number of times word $i$ appears in the context of word $j$.
    + [x] The weighting function $f(.)$ must satisfy $f(0) = 0$.

10. You have trained word embeddings using a text dataset $m_1$ of words. You are considering using these word embeddings for a language task, for which you have a separate labeled dataset $m_2$ of words. Keeping in mind that using word embeddings is a form of transfer learning, under which of these circumstance would you expect the word embeddings to be helpful?
    + **a)** $m_1 >> m_2$
    + b) $m_1 << m_2$