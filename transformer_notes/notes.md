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

Define Query, Key and Value Matricies as $\boldsymbol{Q,K}:(s,d_k)$ and $\boldsymbol{V}:(s,d_v)$.

Write them as below.
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

Define 
$\text{Attention}(\boldsymbol{Q,K,V})=\sigma \left( \dfrac{\boldsymbol{Q K^T}}{\sqrt{d_k}} \right) \boldsymbol{V}$, for row wise softmax $\sigma$.

We see that 
$
\boldsymbol{Z}_{i,j}\coloneqq \boldsymbol{QK^T}_{i,j} = \boldsymbol{q_i \cdotp k_j}
$
is like the similarity score between query $i$ and key $j$.

Rowise softmax means,
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