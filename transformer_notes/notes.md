# Transformer Overview


Input Matrix 
$X \in \mathbb{R}^{s \times d}$. Let's denote as 
$\color{red}{X : (s,d)}$ since it is more readable. We mean s and d to mean sequence length and dimension of model respectively.

# Input Embeddings
The raw text is a sequence of vocabulary terms. Tokenization is the process of splitting the sentence into a sequence of vocabulary terms. Vocab terms can be single words but often tokenizers split the sentences into smaller components than words. In illustrative examples we usually show tokens as words.

## Example
$s=5$, $d$ is usually 512 in the traditional paper.
$$
\begin{align*}
   \text{Sentence}: &
   &\text{THE}     & &\text{WEATHER}   & &\text{IS}    & & \text{LOVELY}   & &\text{TODAY}
   \\
   \downarrow
   \\
   \text{Input IDs}: &
   &\text{105 }    & &\text{102 }      & &\text{12 }   & &\text{24 }   & &\text{12}
   \\
   \downarrow
   \\
   \text{Embedding, }X :(5,d) = [X_{1,},...,X_{5,}]:&
   &X_{1,}      & &X_{2,}   & &X_{3,}   & &X_{4,}   & &X_{5,}
\end{align*}

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

Define Query, Key and Value Matricies as $Q,K:(s,d_k)$ and $V:(s,d_v)$.

Write them as below.
$$
Q=
\begin{bmatrix}
   \mathbf{q_1^T}   \\
   \vdots   \\
   \mathbf{q_s^T} 
\end{bmatrix}

\text{for }
\mathbf{q_i} \in \mathbb{R}^{d_k}, i \in \left\{ 1,...,s\right\}
$$

$$
K=
\begin{bmatrix}
   \mathbf{k_1^T}   \\
   \vdots   \\
   \mathbf{k_s^T} 
\end{bmatrix}

\text{for }
\mathbf{k_i} \in \mathbb{R}^{d_k}, i \in \left\{ 1,...,s\right\}
$$

$$
V=
\begin{bmatrix}
   \mathbf{v_1^T}   \\
   \vdots   \\
   \mathbf{v_s^T} 
\end{bmatrix}

\text{for }
\mathbf{v_i} \in \mathbb{R}^{d_v}, i \in \left\{ 1,...,s\right\}
$$

Define 
$\text{Attention}(Q,K,V)=\sigma \left( \dfrac{\mathbf{Q K^T}}{\sqrt{d_k}} \right)$ for row wise softmax.

We see that 
$
\mathbf{QK^T}_{i,j} = \mathbf{q_i \cdotp k_j}
$
is like the similarity score between query $i$ and key $j$.

Rowise softmax means,
$$
\sigma \left( \mathbf{A} \right)_{i,j} = 
\cfrac{\exp \left( A_{i,j} \right)}{\sum_{r=1}^s \exp \left( A_{i,r} \right)} = 
\cfrac{
   \exp \left( \mathbf{q_i \cdotp k_j} \right)
   }
   {\sum_{r=1}^s \exp \left( \mathbf{q_i \cdotp k_r} \right)
   } = 
\cfrac{\text{score(i,j)}}{\text{row i score}}
\equalscolon \alpha(\mathbf{q_i, k_j}|K)
$$
so we essentially have relative similarity between query $i$ and key $j$