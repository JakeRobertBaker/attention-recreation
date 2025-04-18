# Transformer Overview


Input Matrix 
$X \in \mathbb{R}^{s \times d}$. Let's denote as 
$\color{red}{X : (s,d)}$ since it is more readable. We mean s and d to mean sequence length and dimension of model respectively.

# Input Embeddings
The raw text is a sequence of vocabulary terms. Tokenization is the process of splitting the sentence into a sequence of vocabulary terms. Vocab terms can be single words but often tokenizers split the sentences into smaller components than words. In illustrative examples we usually show tokens as words.

## Example
$s=5$, $d$ is usually 512 in the traditional paper.
$$
\begin{array}{ccc}

\text{Sentence}=
\begin{bmatrix}
\text{THE} \\
\text{WEATHER} \\
\text{IS} \\
\text{LOVELY} \\
\text{TODAY}
\end{bmatrix} &

\mapsto
\quad
\text{Input IDs}=
\begin{bmatrix}
105 \\
12 \\
24 \\
107 \\
57
\end{bmatrix} &

\mapsto
\quad
\text{Embedding, } X: (5,d)=
\begin{bmatrix}
\boldsymbol{x_1^T} \\
\boldsymbol{x_2^T} \\
\boldsymbol{x_3^T} \\
\boldsymbol{x_4^T} \\
\boldsymbol{x_5^T}
\end{bmatrix}

\end{array}
$$

# Positional Encoding
Fixed definition for each position 
$p \in \left\{1,...,s \right\}$
and dimension index 
$i \in \left\{1,...,d \right\}$.

$$
PE :(s,d)  =
\begin{cases} 
PE_{p, 2i}     &=&   \sin \left( \dfrac{p}{10,000^{\left(\dfrac{2i}{d}\right)}} \right)
\\
\\
PE_{p, 2i+1}   &=&   \cos \left( \dfrac{p}{10,000^{\left(\dfrac{2i}{d}\right)}} \right)
\end{cases}
$$

##
Model recieves $X + PE : (s,d)$

# Self Attention
This mechanism is what changed everything.

## Attention Definition

Define Query, Key and Value Matricies as $\boldsymbol{Q,K}:(s,d_k)$ and $\boldsymbol{V}:(s,d_v)$.

Define 
$\text{Attention}(\boldsymbol{Q,K,V}) \coloneqq\sigma \left( \dfrac{\boldsymbol{Q K^T}}{\sqrt{d_k}} \right) \boldsymbol{V}$, for row wise softmax $\sigma$.

## Explanation

Write the matricies as,
$$
\boldsymbol{Q}=
\begin{bmatrix}
   \boldsymbol{q_1^T}   \\
   \vdots   \\
   \boldsymbol{q_s^T} 
\end{bmatrix}

\text{for }
\boldsymbol{q_i} \in \mathbb{R}^{d_k}, i \in \left\{ 1,...,s\right\}
$$

$$
\boldsymbol{K}=
\begin{bmatrix}
   \boldsymbol{k_1^T}   \\
   \vdots   \\
   \boldsymbol{k_s^T} 
\end{bmatrix}

\text{for }
\boldsymbol{k_i} \in \mathbb{R}^{d_k}, i \in \left\{ 1,...,s\right\}
$$

$$
\boldsymbol{V}=
\begin{bmatrix}
   \boldsymbol{v_1^T}   \\
   \vdots   \\
   \boldsymbol{v_s^T} 
\end{bmatrix}

\text{for }
\boldsymbol{v_i} \in \mathbb{R}^{d_v}, i \in \left\{ 1,...,s\right\}
$$


We see that 
$
\boldsymbol{Z}_{i,j}\coloneqq \boldsymbol{QK^T}_{i,j} = \boldsymbol{q_i \cdotp k_j}
$
is like the similarity score between query $i$ and key $j$.

Rowise softmax gives,
$$
\sigma \left( \boldsymbol{Z} \right)_{i,j} = 
\cfrac{
   \exp \left( \frac{1}{\sqrt{d_k}} \boldsymbol{Z}_{i,j} \right)
   }
   {\sum_{r=1}^s \exp \left( \frac{1}{\sqrt{d_k}} \boldsymbol{Z}_{i,r} \right)
   } = 
\cfrac{
   \exp \left( \frac{1}{\sqrt{d_k}} \boldsymbol{q_i \cdotp k_j} \right)
   }
   {\sum_{r=1}^s \exp \left( \frac{1}{\sqrt{d_k}} \boldsymbol{q_i \cdotp k_r} \right)
   } = 
\cfrac{\text{score}(i,j)}{\sum_{r=1}^s \text{score}(i,r)}
\equalscolon 
\alpha(\boldsymbol{q_i, k_j}|\boldsymbol{K})
$$
so we essentially have row normalised similarity between query $i$ and key $j$.

Let 
$\boldsymbol{A} \eqqcolon \text{Attention}(\boldsymbol{Q,K,V})$, then
$$
\boldsymbol{A}_{i,j} 
= \sum_{r=1}^s \sigma \left( \boldsymbol{Z} \right)_{i,r}  \boldsymbol{V}_{r,j}
= \sum_{r=1}^s \alpha(\boldsymbol{q_i, k_r}|\boldsymbol{K}) [\boldsymbol{v_r}]_j
$$

and therefore row $i$ is,
$$
\boldsymbol{A_{i,:}}
= \sum_{r=1}^s \sigma \left( \boldsymbol{Z} \right)_{i,r}  \boldsymbol{V}_{r,j}
= \sum_{r=1}^s \alpha(\boldsymbol{q_i, k_r}|\boldsymbol{K})  \boldsymbol{v_r^T}
\in \mathbb{R}^{1,d_v}
$$
the sum of values, each weighted by query $i$'s similarity with that key.

## Remarks

### Attention is permutation invariant
Notice that row $i$ of $\boldsymbol{A}$ is a function of $\boldsymbol{q_i, K,V}$,
$$
\text{row}_i \left( \boldsymbol{A} \right) =
\sum_{r=1}^s \alpha(\boldsymbol{q_i, k_r}|\boldsymbol{K})  \boldsymbol{v_r^T}
\eqqcolon f(\boldsymbol{q_i, K,V})
$$

Therefore if we permute two rows of A, the output of Attention has the same two rows permuted.

## Muti Head Attention Definition
Let $s$ be sequence length, $d$ be model dimension, $h$ be the number of heads and $d_k=d_v=\frac{d}{h}$.


$$
\begin{align*}
\text{MultiHeadAttention}(\boldsymbol{Q,K,V})
&= [\boldsymbol{H_1},..., \boldsymbol{H_h}]
&&:(s,h\times d_v = d)
\\
\boldsymbol{H_i}
&= 
\text{Attention}
\left( \boldsymbol{QW_i^Q}, \boldsymbol{KW_i^K}, \boldsymbol{VW_i^V} \right)
&&:(s, d_v) 
\end{align*} 
$$

Where matricies

$$
\begin{array}{ccc}
\boldsymbol{W^Q} =
[\boldsymbol{W_1^Q} ,...,\boldsymbol{W_1^Q}], &
\boldsymbol{W^K} =
[\boldsymbol{W_1^K} ,...,\boldsymbol{W_1^K}], &
\boldsymbol{W^Q} =
[\boldsymbol{W_1^V} ,...,\boldsymbol{W_1^V}]
\end{array}
:(d,d)
$$

are learnable parameters with 
$\boldsymbol{W_i^Q},\boldsymbol{W_i^K},\boldsymbol{W_i^V}:(d,d_V)$ for 
$i=1,...,h$.

This is essentially a trick to do all the compute in one go.

$$
\begin{align*}
\boldsymbol{QW^Q}
&= \boldsymbol{Q} [\boldsymbol{W_1^Q} ,...,\boldsymbol{W_1^Q}]
\\
&= [\boldsymbol{Q} \boldsymbol{W_1^Q} ,...,\boldsymbol{Q} \boldsymbol{W_1^Q}]
\end{align*}
$$