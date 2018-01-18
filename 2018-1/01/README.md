
# Multimodal Word Distributions

* Abstract 摘要

## 1. Introduction
 

## 2. Related Work

## 3. Methodology

### 3.1 Word Representation

### 3.2 Skip-Gram
训练对象的学习参数$\theta = \lbrace {\overrightarrow{u_{w,i}}, p_{w,i}, \sum{_{w,i}} \rbrace }$ 的想法来自于连续skip-gram模型(Mikolov et al., 2013a), 在skip-gram模型中训练单词嵌入（word embedings)的目的是为了最大化一个单词与其相邻单词的概率。这个过程遵循了在自然语境中单词的出现是语义相关的这样一种分布式假设。举例：单词'jazz'和'music'出现的趋势会比'jazz'和'cat'出现的趋势更平常。因此'jazz'和'music'更有可能相关。单词表示的学习包含useful semantic information和可以用来执行自然语言处理的各种任务。例如，单词的相似性分析，语义分类，单词类比建模或者作为一个复杂系统（统计机器翻译）的预处理输入。

<div align="center">
<img src="images/figure.png" height="50%" width="50%">
</div>

上面的Figure 1 是一个高斯混合嵌入的模型，每一个组件都关联一个不同的意思，每一个高斯组件表示为一个椭圆形，椭圆形的中心是均值向量，表面的值线表示协方差矩阵，这个矩阵反应了均值和不确定性的微妙变化。图的左半部分是单词的高斯混合分布，在这个分布中高斯组件随机地初始化。训练之后，在图的右侧可以看到单词'rock'组件更加接近于'stone' 和'basalt', 其他高斯组件也有同样的现象:'music'更接近于'jazz'和'pop'。同时也证明了逻辑蕴含的概念，在逻辑蕴含中更一般化的单词'music'就会被'jazz', 'pop', 'rock'这些单词修饰。在图的下面是高斯嵌入模型（Vilnis and McCallum, 2014)。单词具有多层意思，比如'rock', 学习表示的方差没必要地变大就是为了给其他含义分配一些概率。此外，这些词的均值向量可以在两个聚类之间进行，将分布的质量集中在远离某些含义的区域上。

### 3.3 Energy-based Max-Margin Objective
对象中的每个实例都包含两对单词，$\left ( w, c \right )$ 和 $\left (w, {c}' \right )$, $w$是从语料的句子中抽样的，$c$是距离这个单词长度为$l$的上下文中的单词。举例：单词$w$=jazz在句子 "I listen to the jazz music"的上下文单词有： I, listen, to, music。${c}'$是随机抽取的非上下问单词，例如 ${c}'$ = ariplane。

我们的目标是最大化单词$w$和$c$出现的energy（注：文章中是energy，以我的思维翻译为概率，但是作者没有用概率哪个单词，而是用了这个。）, 最小化$w$和${c}'$出现的energy。这个过程和负抽样（negative sampling Mikolov el al., 2013a, b)很相似, 通过对比正上下文对和负上下文对之间的点积。energy函数是对分布和3.4节中讨论的内容的相似性的度量。

我们使用了 max-margin 排序目标（Joachims, 2002),这在Vilnis和McCallum（2014）的高斯嵌入中使用了这种方法。通过计算m可以使得一个单词的正上下文语境的相似性高于负上下文语境的相似性。

$$L_{\theta}(w,c,{c}') = max(0, m-logE_\theta(w,c) + logE_\theta(w,{c}'))$$

关于$\theta=\{\overrightarrow{u_{w,i}},p_{w,i}, \sum_{w,i} \}$的随机梯度下降算法可以最小化上面的$L_{\theta}$。


**单词采样** 用和word2vec（Mikolov, et al., 2013a,b)的实现很相似的单词采样来平衡frequent words和rare words的重要性。Frequent words比如 'the', 'a', 'to'，这些单词相对于frequent word 'dog', 'love', 'rock'这些单词没有太大的意义，有时候可能更加关注出现频率较少的具有语义的单词。我们使用子采样来改善学习单词向量的性能（Mikolov et al., 2013b). 这里对单词$w_{i}$以$P(w_{i})=1-\sqrt(t/fw_{i})$, 公式中$f(w_{i})$是单词$w_{i}$在训练预料中国呢出现的次数，$t$是一个频率的阈值。

为了产生负上下文单词，每个单词$w_{i}$输入后根据分布$P_{n}\propto U(w_{i})^{3/4}$， 这是一个均匀分布的扭曲的版本，经常被用来消除frequent words的相对重要性。 子采样和负分布的选择在word2vec中都被证明是有效的。

### 3.4 Energy Function

在相似性计算（energy function)中，单词经常被表示为向量，相似性的计算就表示为两个向量的点积。 我们的用单词的分布来代替向量，因此我们需要一种度量，这种度量不仅要体现相似性，还要体现无关性。

我们提出了使用 expected likelihood kernel.它是向量之间的内机的分布之间的内积，是对（jebara et al., 2004)的一个扩展。

$$ E(f,g)=\int{f(x)g(x)dx} = <f,g>L_{2}$$

公式中$<.,.>L_{2}$表示Hilbert空间$L_{2}$的内积。我们选择这种形式的energy主要是因为它可以用接近的形式进行评估，由于我们在公式（1）中概率嵌入的选择。

<div align="center">
<img src="images/fig002.png" height="50%", width="50%">
</div>


## 4. Experiments

### 4.1 Hyperparameters

### 4.2 Similarity Measures

#### 4.2.1 Expected Likelihood Kernel

#### 4.2.2 Maximum Cosine Similarity

#### 4.2.3 Minimum Euclidean Distance

### 4.3 Qualitative Evaluation

### 4.4 Word Similarity

### 4.5 Word Similarity for Polysemous Words

### 4..6 Reduction in Variance of Polysemous Words

### 4.7 Word Entailment
 

## 5 Discussion

