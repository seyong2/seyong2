---
layout: post
title: Understanding Transformers
subtitle: Breaking Down the Transformer Architecture
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [Artificial Intelligence, Data Science, Deep Learning, Transformer, Attention]
comments: true
---

In this post, we will explore how **Transformer**, the basis of ChatGPT, works one step at a time with a simple example. Specifically, we'll focus on how a transformer neural network translates a simple Spanish sentence "*Te quiero*" into English, "*I love you*".

# 1. Word Embedding

We first need to convert the words into numbers as neural networks cannot receive words as input and a transformer is a type of neural network. For this purpose, we'll use word embedding, and the main idea of word embedding is to use a simple neural network that allows us to have a numerical representation of text that captures the semantic meaning of words. Suppose we have a vocabulary consisting of 5 tokens *<SOS>* (Start of Sentence), *te*, *y*, *quiero*, and *vas*. We want the network to produce two values for each token. In other words, the number of dimensions for word embeddings is set to 2. Then, the word embedding model will look like below. We use identity activation functions, meaning that the output values are the same as the input values. Then, starting from a one-hot matrix where each row has a value of 1 for each token and 0s for the other ones, as our input tokens are *<SOS>* (Start of Sentence), *te*, and *quiero*, the resulting word embedding vector will have a matrix with a shape of (3 $\times# 2).

![word_emb](https://github.com/user-attachments/assets/76a93ed3-a74b-4770-8d18-dcefd2df45e8)

$$ A * W_{emb} = {\left\lbrack \matrix{1 & 0 & 0 & 0 & 0 \cr 0 & 1 & 0 & 0 & 0 \cr 0 & 0 & 0 & 1 & 0} \right\rbrack} * \left\lbrack \matrix{w_{00} & w_{01} \cr w_{10} & w_{11} \cr w_{20} & w_{21} \cr w_{30} & w_{31} \cr w_{40} & w_{41} \cr w_{50} & w_{51}} \right\rbrack = \left\lbrack \matrix{w_{00} & w_{01} \cr w_{10} & w_{11} \cr w_{40} & w_{41}} \right\rbrack $$

# 2. Position Encoding

The next step is to add a set of numbers that correspond to word order to the embedding values for each token. As we do not use sequential neural networks like RNNs or LSTMs, the word embedding vectors do not contain any information about the order. However, there are many situations where the order of the words is very important. Those numbers that assign orders come from a sequence of alternating sine and cosine functions. More specifically, for each position $p$ of token in the input, and each word embedding dimension $2i$ (even-indexed) and $2i+1$ (odd-indexed) in the encoding vector...

$$PE(p, 2i)=sin(\frac{p}{10000^{\frac{2i}{d}}})$$

$$PE(p, 2i+1)=cos(\frac{p}{10000^{\frac{2i}{d}}})$$

where $d$ is the output embedding space and in this example, it's equal to 2.

Therefore, the position encoding matrix has a shape of (3 $\times$ 2) and it's equal to 

$$ PE= \left\lbrack \matrix{sin(\frac{0}{1000^{2\times0/2}}) & cos(\frac{0}{1000^{2\times0/2}}) \cr sin(\frac{21}{1000^{2\times0/2}}) & cos(\frac{1}{1000^{2\times0/2}}) \cr sin(\frac{2}{1000^{2\times0/2}}) & cos(\frac{2}{1000^{2\times0/2}})} \right\rbrack = \left\lbrack \matrix{0 & 1 \cr 0.84 & 0.54 \cr 0.91 & -0.42} \right\rbrack $$

The sine and cosine functions get wider for larger embedding positions. Then, we do the element-wise addition, $A*W_{emb}+PE$, to get each token's word embeddings plus positional encodings.

# 3. Self-Attention

Self-attention is a mechanism in neural networks that allows us to see how tokens interact with each other. Each token looks at other tokens in the input sentence with an attention mechanism, gathers context, and updates the previous representation of self. This concept is implemented with query-key-value attention.

- The query is used when a token looks at others. It's seeking the information to understand itself better. We multiply the encoded values by the transposed query weights matrix ($W_Q^T$) whose shape is 2 $\times$ 2. Let's call the resulting 3 $\times$ 2, $Q$.

- The key is responding to a query's request. It's used to compute attention weights. As before, we do matrix multiplication between the encoded values and the transpose of key weights matrix ($W_K^T$) to get $K$.

- The value is used to compute attention output: it gives information to the tokens which "say" they need it. To get $V$, we multiply the encoded values by $W_V^T$, the transposed value weights matrix. 

To compute the similarities between the words, that is, between the query and the keys, we use dot product, $Q * K^T$. The words that are highly associated with each other will have large values. Then, as this dot product produces unscaled similarity scores, we divide $Q * K^T$ by $\sqrt{d_k}$ where $d_k$ is the dimension of $K$ to get scaled similarities. Next, we apply the softmax function to each row of the matrix of scaled similarity scores so that the values of each row sum up to 1. We can think of these values as a way to determine what percentage of each input word we should use to encode itself and the other words. Finally, we need to calculate attention by multiplying these percentages by $V$. The self-attention score matrix will have a (3 $\times$ 2) shape.

In this example, we are using only a self-attention cell, but we can use multiple self-attention cells with their own sets of weights in order to capture better how words are related in complicated sentences (Multi-Head Attention).

# 4. Residual Connection

We take the position encoded values and add them to the self-attention values. These bypasses are called residual connections, and they make it easier to train complex neural networks by allowing the self-attention layer to establish relationships among the input words without having to also preserve the word embedding and position encoding informaiton. As both matrices are of the same shape, the resulting enocded values will also have a shape of (3 $\times$ 2).

Now that we've encoded the Spanish input phrase, it's time to decode it to English, "*I love you*".

# 5. Word Embedding

Just like the encoder, the decoder starts with word embedding for the English vocabulary and let's suppose that the vocabulary consists of 6 words, *<SOS>* (to initialize the decoder), *I*, *love*, *you*, *and*, and *<EOS>* (to stop generating output). To make training go faster, we implement teacher forcing by initializing the decoders with known output values during training. Then, we initialize the first decoder with *<SOS>* token, the second decoder with *I*, the third one with *love*, and the final decoder with *you* which will output *<EOS>* token. Thus, our decoder has 4 tokens and the resulting word embedding vector is a (4 $\times$ 2) matrix.

matrix visual

# 6. Position Encoding

Just like before we did in the encoder, we add the position encoding to the word embedding matrix.

# 7. Masked Self-Attention

Calculating self-attention scores in the decoder starts out exactly like it does in the encoder. In the decoder, self-attention is a bit different from the one in the encoder. While the encoder receives all tokens at once and the tokens can look at all tokens in the input sentence, in the decoder, we generate one token at a time: during generation, we don't know which tokens we'll generate in future. For example, at the start of decoding, we only use the similarity between *<SOS>* and itself because we don't want tokens to cheat and peak at what comes next when calculating self-attention.

To forbid the decoder to look ahead, the model uses masked self-attention: future tokens are masked out. The way that transformers keep track of what values should and shouldn't be used to compute self-attention for each token is to take the scaled similarity score matrix and add a matrix to it. The mask matrix adds 0s to values we want to include in the attention calculations and -infinity to any value that we need to ignore. Then, when applied to the softmax function, the -infinity will have 0, and no influence on the encoding of the words that come before.

# 8. Residual Connection

We take the self-attention scores and add the original word and position encodings to each token.

# 9. Encoder-Decoder Attention

The calculation is the same as calculating standard self-attention but the $V$ and $K$ matrices are created with the Encoder's output matrix and the $Q$ matrix is created with the values from the Decoder.

# 10. Residual Connection

When obtaining the encoder-decoder attention scores, we add them to the next set of residual connections.

# 11. Fully-Connected Layer

It has two inputs for the two values we have computed for each token and six outputs, one for each token in the output vocabulary. Then the fully-connected layer weights matrix has (2 $\times$ 5) shape. When multiplying the new decoder values and the weights, we add bias terms to each column.

# 12. SoftMax

We run the output from the fully-connected layer through a softmax function from which we get the prediction for the next word.

### References
 
- [Transformer: Attention is All You Need](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#transformer_intro)
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=20)
- [The matrix math behind transformer neural networks, one step at a time!!!](https://www.youtube.com/watch?v=KphmOJnLAdI&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=25)
