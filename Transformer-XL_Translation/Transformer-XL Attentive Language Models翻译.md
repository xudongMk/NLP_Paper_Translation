## Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

**Zihang Dai***∗*12**, Zhilin Yang***∗*12**, Yiming Yang**1 **, Jaime Carbonell**1**,**

**Quoc V. Le**2 **, Ruslan Salakhutdinov**1 

1Carnegie Mellon University, 2Google Brain

{dzihang,zhiliny,yiming,jgc,rsalakhu}@cs.cmu.edu, qvl@google.com

### 摘要

Transformers 具有学习长期依赖的潜力，但在语言模型设置中受到固定长度的上下文的限制。我们提出了一种新的语言模型 Transformers-XL，它可以在不破坏时间一致性的情况下，使学习依赖关系超出固定长度的限制。它包括一种分段recurrence机制和一种新的位置编码方案。我们的方法不仅能够捕获长期依赖关系，而且还解决了上下文固定长度问题。Transformer-XL学习的依赖关系比RNNs长80%，比Transformers长450%，在短序列和长序列上都能获得更好的性能，在评估期间比普通的Transformers快1800多倍。值得注意的是，我们将enwiki8上的bpc/perplexity的最新结果达到了0.99，text8达到了1.08，WikiText-103达到了18.3，Billion Word达到21.8，Penn Treebank达到54.5（无微调）。当只在WikiText-103上进行训练时，Transformer XL设法生成具有数千个标记的合理连贯、新颖的文本文章。我们的代码、预训练模型和超参数可以同时在Tensorflow和PyTorch中【https://github.com/kimiyoung/transformer-xl】使用。

### 1 简介

语言模型是最重要的问题之一，需要建模长期依赖性，成功的应用如无监督预训练(Dai and Le, 2015;Peters 等，2018;Radford 等，2018;Devlin 等，2018)。然而，如何使神经网络具备对序列数据进行长期依赖建模的能力一直是一个挑战。递归神经网络(RNNs)，特别是长短期记忆(LSTM)网络(Hochreiter and Schmidhuber, 1997)已经成为语言建模的标准解决方案，在多个基准测试上取得了较好的结果。尽管RNNs具有广泛的适应性，但由于梯度消失和爆炸，RNNs很难优化(Hochreiter 等， 2001)，而在LSTMs中引入门控和梯度剪切技术(Graves, 2013)可能不足以完全解决这个问题。根据经验，先前的研究发现，LSTM语言模型平均使用200个上下文单词(Khandelwal et al.， 2018)，这表明还有进一步改进的空间。

另一方面，在注意机制中远距离单词对之间的直接连接可能会更易于优化，并能够学习长期依赖(Bahdanau 等， 2014;Vaswani 等，2017)。最近，Al-Rfou 等人(2018)设计了一套辅助损失，用于训练字符级语言建模的深度 Transformer 网络，其性能大大优于 LSTMs。尽管取得了成功，Al-Rfou 等人（2018）的 LM 训练是把几百个字符分隔成固定长度的片段上进行的，没有任何跨段的信息流。由于固定的上下文长度，模型不能捕获任何超出预定义上下文长度的长期依赖关系。此外，通过选择连续的符号块来创建固定长度的段，而不考虑句子或任何其他语义边界。因此，该模型缺乏必要的上下文信息来很好地预测前几个符号（应该指的是每段的前几个符号），导致优化不足和性能低下。我们将这个问题称为上下文碎片（context fragmentation）问题。

为了解决前面提到的固定长度上下文的限制，我们提出了一个名为Transformer-XL(意为超长)的新架构。我们将递归的概念引入我们深层 self-attention 网络。特别是，我们重用了前面片段获得的隐藏状态，而不是从头计算每个新片段的隐藏状态。被重用的隐藏状态充当当前片段的记忆，这在片段之间构建了循环连接。因此，建模非常长期的依赖关系成为可能，因为信息可以通过循环连接传播。同时，之前片段的信息也能解决上下文碎片化问题。更重要的是，我们展示了使用相对位置编码而不是绝对位置编码的必要性，以便在不造成时序混乱的情况下实现状态重用。因此，作为一项额外的技术贡献，我们引入了一个简单但更有效的相对位置编码公式，该公式能推广到比训练时看到的更长的注意力长度。

Transformer-XL在五个数据集上获得了很好的结果，这些数据集从单词级到字符级各不相同。在仅 100M tokens 上训练的 Transformer-XL 还能够生成具有数千个 tokens 的相对连贯的长文本文章。

我们的主要技术贡献包括在一个纯自注意力模型中引入递归的概念，并推导出一种新的位置编码方案。这两种技术构成了一组完整的解决方案，因为其中任何单独一种都不能解决固定长度上下文的问题。Transform-XL 是第一个在字符级和单词级语言建模方面都比 RNNs 取得更好效果的自注意力模型。

### 2 相关工作

在过去的几年里，语言建模领域见证了许多重大的进步，包括但不限于设计新的架构来更好地编码上下文(Bengio 等， 2003;Mikolov 等，2010;Merity 等，2016;Al-Rfou 等，2018)，改进正则化和优化算法(Gal 和 Ghahramani,2016)，加快 Softmax 计算(Grave 等，2016)，丰富输出分布家族(Yang 等，2017)。

为了在语言建模中捕获远程上下文，一直以来的方式是直接将更广泛上下文的表征作为附加输入输入到网络中。目前的工作从手动定义上下文表征到依赖从数据中学到的文档级主题。

更广泛地说，在通用序列建模中，如何捕获长期依赖性一直是一个长期存在的研究问题。从这个角度来看，自 LSTM 的普适性以来，人们一直致力于缓解梯度消失问题,包括更好的初始化(Le 等, 2015),额外损失信号(Trinh 等, 2018)，增强记忆结构（Ke 等，2018）和其他修改RNNs内部结构以更好优化。与之不同的是，我们的工作是基于Transformer的体系结构，并表明作为现实世界的任务的语言建模能够从学习长期依赖关系中获益。

### 3 模型

给出一个 tokens 为 $\mathrm{x}=(x_1,\cdots,x_T)$ 的语料，语言模型的任务是估计联合概率 $P(\mathrm{x})$,通常被自回归因子分解为 $P(\mathrm{x})=\prod_tP(x_t|\mathrm{x}_{<t})$。通过因子分解，问题简化为估计每个条件因子。在这项工作中，我们坚持使用标准的神经方法来建模条件概率。具体来说，使用一个可训练的神经网络将上下文$\mathrm{x}_{<t}$编码成一个固定大小的隐藏状态，并与词嵌入相乘得到 logits，然后将 logits 输入 Softmax 函数，生成关于下个 token 的类别的概率分布。

#### 3.1 普通的Transformer模型

为了将 Transformer 或 self-attention 应用到语言建模中，核心问题是如何训练 Transformer 将任意长的上下文有效地编码为固定大小的表征。假定给无限内存和计算量，一个简单的解决方案就是使用绝对的 Transformer 解码器(类似于前馈神经网络)处理整个上下文序列。然而，在实践中，由于资源有限，这通常是不可行的。

一个可行但粗糙的近似方法是将整个语料库分割成易处理大小的较短片段，仅仅用每个片段来训练模型，忽略前面片段中的所有上下文信息。这是Al-Rfou等人(2018)采用的方法。我们称其为普通的Transformer模型，如图1中a部分所示。但是在这种训练模式下，信息在向前或向后的传递过程中都不会跨段流动。使用固定长度上下文有两个关键限制。首先，最大的依赖长度是以片段长度为上界的，在字符级语言建模中，片段长度可能只有几百（Al-Rfou 等，2018）。虽然 self-attention 机制相对于 RNNs 受到梯度消失的影响更小，但普通的语言模型不能充分利用这种优化优势。其次，虽然可以用 padding 来关注句子或者其他语义边界，但实际中，为了提高效率，简单地将长文本分割成固定长度的片段已经成为了标准的做法(Peters 等，2018;Devlin 等，2018;Al-Rfou等，2018)。然而，简单地将一个序列分成固定长度的片段将导致第1节中讨论的上下文碎片问题。

在评估过程中，每一个步骤都只取和训练过程一样的固定长度，但只在最后一个位置做了一次预测。然后，在下一步中，该段只向右移动一个位置，新段必须从头开始处理。如图1中的b部分。该过程确保了每个预测都利用了训练中暴露的尽可能长的上下文信息，同时也缓解了训练中遇到的上下文碎片问题。然而，这个评估过程的代价是非常昂贵。而我们提出的架构能够显著提高评估速度。

![图1](E:\python_work_space\NLP_Paper_Translation\Transformer-XL_Translation\pic\fig1.png)

#### 3.2 状态重用的片段递归模型

为了解决使用固定长度上下文的局限性，我们在 Transformer 架构中引入一种递归机制。在训练过程中，对前一片段计算得到的隐藏状态序列进行固定并缓存，当模型处理下一个新的片段时，将其作为扩展的上下文进行重用，如图2中的a部分。尽管梯度仍然保留在一个片段中，但是这个额外的输入允许网络利用历史信息，从而能够对长期依赖关系建模并避免上下文碎片化。从公式来看，分别让两个长度为 $L$ 的连续分段为 $\mathrm{s}_{\tau} = [x_{\tau,1},\cdots,x_{\tau,L}]$ 和 $\mathrm{s}_{\tau+1} = [x_{\tau+1,1},\cdots,x_{\tau+1,L}]$。用$\mathrm{h}_{\tau}^n \in \mathbb{R}^{L\times d}$ 来表示第 $n$ 层第 $\tau$ 段 $\mathrm{s}_{\tau}$ 序列的隐藏状态，$d$ 是隐藏维度。然后，用下面公式生成分段$\mathrm{s}_{\tau+1}$的第$n$层隐状态：
$$
\begin{aligned}

& \widetilde{\mathrm{h}}_{\tau+1}^{n-1} = [\mathrm{SG}(\mathrm{h}_{\tau}^{n-1}) \circ \mathrm{h}_{\tau+1}^{n-1}] \\

& \mathrm{q}_{\tau+1}^n,\mathrm{k}_{\tau+1}^n,\mathrm{v}_{\tau+1}^n = \mathrm{h}_{\tau+1}^{n-1} \mathrm{W}_q^{\top},\widetilde{\mathrm{h}}_{\tau+1}^{n-1} \mathrm{W}_k^{\top},\widetilde{\mathrm{h}}_{\tau+1}^{n-1}\mathrm{W}_v^{\top} \\

& \mathrm{h}_{\tau+1}^n=\mathrm{Transformer-Layer}(\mathrm{q}_{\tau+1}^n,\mathrm{k}_{\tau+1}^n,\mathrm{v}_{\tau+1}^n)

\end{aligned}
$$
其中，函数$\mathrm{SG}(·)$代表停止梯度，符号$[h_u \circ h_v]$表示沿两个隐藏序列在长度维度上拼接，$\mathrm{W}_·$表示模型参数。与标准 Transformer 相比，关键的区别在于 $\mathrm{k}_{\tau+1}^n$ 和 $\mathrm{v}_{\tau+1}^n$ 取决于扩展上下文 $\widetilde{\mathrm{h}}_{\tau+1}^{n-1}$，因此上一段的$\mathrm{h}_{\tau}^{n-1}$被缓存。 我们通过图2中a部分中的绿色路径来强调这个特殊的设计。

将这种递归机制应用于语料库的每两个连续段，本质上是在隐藏状态创建了分段级递归。因此，利用的有效上下文可以远远超出两个分段。然而，值得注意的是$\mathrm{h}_{\tau+1}^n$ 和 $\mathrm{h}_{\tau}^{n-1}$ 之间的循环依赖每段向下移动了一层，这与传统 RNN-LMs 中相同层循环不同。因此，最长的可能依赖长度随着层数和片段长度线性增加，也就是 $O(N \times L)$，如图2中b部分阴影区域所示。这类似于截断的BPTT (Mikolov 等，2010)（back-propagation through time），一种用于训练 RNN-LMs 的技术。但是，与截断的BPTT不同，我们的方法缓存的是一系列隐藏状态，而不是最后一个状态，并且与 3.3 节中描述的相对位置编码技术一起使用。

除了实现超长上下文依赖和解决上下文碎片问题外，循环机制带来的另一个好处是评估速度显著加快。具体地说，在评估期间，可以重用前面片段的表示，而不是像普通Transformer模型那样从头计算。在 enwik8 数据集上，Transformer-XL 在评估期间比普通Transformer模型快1800多倍（详细见第4节）。

最后，值得注意的是循环机制不需要仅限于前一个段。理论上，只要GPU内存允许，我们可以缓存尽可能多的片段信息，并在处理当前段时将它们全部作为额外上下文重新使用。因此，我们可以缓存一个预先定义的长度为 $M$ 的、尽可能跨越多个段的隐藏状态，由于与记忆增强神经网络（Graves 等，2014；Weston 等，2014）有明确的联系，将它们称为 $\mathrm{m}_\tau^n \in \mathbb{R}^{M \times d}$。在实验中，训练期间我们将 $M$ 设置为单个片段的长度，并在评估期间将其增加多倍。

![图2](E:\python_work_space\NLP_Paper_Translation\Transformer-XL_Translation\pic\fig2.png)

#### 3.3 相对位置编码

虽然我们发现上一小节中提出的想法非常吸引人，但是为了重用隐藏状态，我们还y有一个关键的技术挑战没有解决。就是，当我们重用状态时，如何保持位置信息的一致性？回想一下，在标准 Transformer 中，序列顺序信息由一组位置编码提供，记为 $\mathrm{U} \in \mathbb{R}^{L_{\mathrm{max}} \times d}$。其中第 $i$ 行 $\mathrm{U}_i$ 对应一句话（就是一个分段）中第 $i$ 个绝对位置，$L_{\mathrm{max}}$ 规定了要建模的最大可能长度。然后，Transformer 的实际输入是词嵌入和位置编码在按位置的和。如果我们简单地将这个位置编码与我们的递归机制相适应，则隐藏状态序列将由下式计算：
$$
\begin{aligned} & \mathrm{h}_{\tau+1} = f(\mathrm{h}_\tau,\mathrm{E}_{\mathrm{s}_{\tau+1}}+\mathrm{U}_{1:L}) \\ & \mathrm{h}_{\tau} = f(\mathrm{h}_{\tau-1},\mathrm{E}_{\mathrm{s}_{\tau}}+\mathrm{U}_{1:L})

\end{aligned}
$$


其中 $\mathrm{E}_{\mathrm{s}_{\tau}} \in \mathbb{R}^{L \times d}$ 代表序列 $\mathrm{s}_\tau$的词嵌入，$f$ 表示转换函数。注意，$\mathrm{E}_{\mathrm{s}_{\tau}}$ 和 $\mathrm{E}_{\mathrm{s}_{\tau+1}}$ 都与相同的位置编码 $\mathrm{U}_{1:L}$ 有关。因此，对任何$j=1,\cdots,L$，模型都没有信息去区分 $x_{\tau,j}$ 和 $x_{\tau+1,j}$ （第$\tau$段第$j$个字和第$\tau+1$段第$j$个字）的位置差别，导致完全的性能损失。

为了避免这种失败的模型，其基本思想是只对隐藏状态中的相对位置信息进行编码。从概念上讲，位置编码为模型提供了关于如何收集信息的时间线索或“偏差”，即，去哪里学习。为了达到同样的目的，我们不是在初始 embedding 中静态地加入偏差，而是将相同的信息注入到每一层的注意力评分中。更重要的是，以一种相对的方式来定义时间偏差更直观、更通用。例如，当查询向量 $\mathrm{q}_{\tau,i}$ 关注 key 向量 $\mathrm{k}_{\tau,\le i}$ 时，不需要知道每个 key 向量的绝对位置来确定分段（句子）的时间顺序，它就可以知道 每个 key 向量 $\mathrm{k}_{\tau,j}$和它自己 $\mathrm{q}_{\tau,i}$的距离，即 $i-j$。实际上，我们可以创建一组相对位置编码 $\mathrm{R} \in \mathbb{R}^{L_{\mathrm{max}} \times d}$,其中第 $i$ 行 $\mathrm{R}_i$表示两个位置的相对距离$i$。通过动态地将相对距离注入注意力得分中，query 向量可以很容易地通过 $x_{\tau,j}$ 和 $x_{\tau+1,j}$ 的不同距离区分他们,使得状态重用机制可行。同时，我们不会丢失任何时间信息，因为绝对位置可以从相对距离递归地恢复。

在此之前，相对位置编码的概念已经在机器翻译（Shaw 等，2018）和音乐生成（Huang 等，2018）的背景下探索过。在这里，我们提供了一个不同的推导，得出了一种新形式的相对位置编码，它不仅与绝对位置编码有一一对应的关系，而且在经验上也有更好的泛化（见第4节）。首先，在标准 Transformer 中，同一段的内 $q_i$ 和 $k_j$的注意力得分可以分解为：
$$
\begin{aligned}

\mathrm{A}_{i,j}^{\mathrm{abs}} & = q_i^\top k_j \\ &= (\mathrm{E}_{x_i}+\mathrm{U}_{i})^\top \mathrm{W}_q^\top \mathrm{W}_k (\mathrm{E}_{x_j} + \mathrm{U}_{j})\\ &= \underbrace{\mathrm{E}_{x_i}^\top \mathrm{W}_q^\top \mathrm{W}_k \mathrm{E}_{x_j}}_{(a)}

- \underbrace{\mathrm{E}_{x_i}^\top \mathrm{W}_q^\top \mathrm{W}_k \mathrm{U}_{j}}_{(b)} \\ &+ \underbrace{\mathrm{U}_{i}^\top \mathrm{W}_q^\top \mathrm{W}_k \mathrm{E}_{x_j}}_{(c)}
- \underbrace{\mathrm{U}_{i}^\top \mathrm{W}_q^\top \mathrm{W}_k \mathrm{U}_{j}}_{(d)}

\end{aligned}
$$


根据只依赖相对位置信息的思想，我们建议按照如下方式重新构建这四个术语：
$$
\begin{aligned}

\mathrm{A}_{i,j}^{\mathrm{rel}}

&= \underbrace{\mathrm{E}_{x_i}^\top \mathrm{W}_q^\top \mathrm{W}_{k,E} \mathrm{E}_{x_j}}_{(a)}

- \underbrace{\mathrm{E}_{x_i}^\top \mathrm{W}_q^\top \mathrm{W}_{k,R} \mathrm{R}_{i-j}}_{(b)} \\ &+ \underbrace{u^\top\mathrm{W}_{k,E} \mathrm{E}_{x_j}}_{(c)}
- \underbrace{v^\top \mathrm{W}_{k,R} \mathrm{R}_{i-j}}_{(d)}

\end{aligned}
$$

- 第一个变化是，我们将 $(b)$ 和 $(d)$ 中 key 向量的绝对位置嵌入$\mathrm{U}_j$ 全部替换为相对的 $\mathrm{R}_{i-j}$。这在本质上反应了先验，即只有相对距离才会影响到要关注的地方。注意，$R$ 是一个没有可学习参数的正弦编码矩阵（Vaswani 等，2017）。
- 第二是，我们引入了一个可训练参数 $u \in \mathbb{R}^d$ 来代替 $(c)$项的 query $\mathrm{U}_i^\top \mathrm{W}_q^\top$。在这种情况下，由于所有查询位置的查询向量都是相同的，因此，无论查询位置如何，对不同单词的关注偏差都应该保持不变。同理，在$(d)$项中用可训练参数 $v \in \mathbb{R}^d$ 代替 $\mathrm{U}_i^\top \mathrm{W}_q^\top$。
- 最后，为了生成基于内容的 key 向量和基于位置的 key 向量，我们故意将两个权值矩阵 $\mathrm{W}_{k,E}$ 和 $\mathrm{W}_{k,E}$ 分离出来。

在新的参数下，每一项都有直观的含义：$(a)$ 表示基于内容的寻址，$(b)$ 捕获了与内容相关的位置偏差，$(c)$ 控制全局内容偏差，$(d)$ 编码了全局位置偏差。

相比之下，Shaw 等人（2018）的公式只有$(a)$，$(b)$ 项，没有 $(c)$,$(d)$ 两个偏差。此外，Shaw 等人（2018）把乘法$\mathrm{W}_k\mathrm{R}$合并成一个单一的可训练的矩阵 $\hat{\mathrm{R}}$，丢掉建立在原始正弦位置编码（Vaswani 等，2017）的归纳偏置（归纳偏置是个专有名词，可理解为“前提假设”的意思）。相比之下，我们的相对位置嵌入 $\mathrm{R}$ 采用了正弦公式。作为归纳偏置的一个优点，在一定长度记忆上训练的模型，可以在评估过程中自动地推广至原来长度的几倍。

将递归机制与我们提出的相对位置嵌入相结合,我们最终得到了 Transformer-XL 结构。为了完整起见，我们再这里总结了具有单个注意力头的 $n$ 层 Transformer-XL 的计算过程。对于$n=1,\cdots,N$:
$$
\begin{aligned}

\widetilde{\mathrm{h}}_\tau^{n-1} =&[\mathrm{SG}(\mathrm{m}_\tau^{n-1}) \circ \mathrm{h}_\tau^{n-1}] \\

\mathrm{q}_\tau^n,\mathrm{k}_\tau^n,\mathrm{v}_\tau^n =& \mathrm{h}_\tau^{n-1} {\mathrm{W}_q^n}^\top,\widetilde{\mathrm{h}}_\tau^{n-1} {\mathrm{W}_{k,E}^n}^\top,\widetilde{\mathrm{h}}_\tau^{n-1} {\mathrm{W}_{v}^n}^\top \\

\mathrm{A}_{\tau,i,j}^n =& {\mathrm{q}_{\tau,i}^n}^\top \mathrm{k}_{\tau,j}^n + {\mathrm{q}_{\tau,i}^n}^\top \mathrm{W}_{k,R}^n \mathrm{R}_{i-j} \\

&+u^\top \mathrm{k}_{\tau,j} + v^\top\mathrm{W}_{k,R}^n \mathrm{R}_{i-j} \\

\mathrm{a}_\tau^n =& \mathrm{Masked-Softmax}(\mathrm{A}_\tau^n)\mathrm{v}_\tau^n \\

\mathrm{o}_\tau^n =& \mathrm{LayerNorm}(\mathrm{Linear}(\mathrm{a}_\tau^n)+\mathrm{h}_\tau^{n-1}) \\

\mathrm{h}_\tau^n =& \mathrm{Positionwise-Feed-Forward}(\mathrm{o}_\tau^n)

\end{aligned}
$$
(注意：论文中的公式$\mathrm{A}_{\tau,i,j}^n$应该是少打印了个$n$，应该是
$$
\begin{aligned} \mathrm{A}_{\tau,i,j}^n =& {\mathrm{q}_{\tau,i}^n}^\top \mathrm{k}_{\tau,j}^n + {\mathrm{q}_{\tau,i}^n}^\top \mathrm{W}_{k,R}^n \mathrm{R}_{i-j} \\

&+u^\top \mathrm{k}_{\tau,j}^n + v^\top\mathrm{W}_{k,R}^n \mathrm{R}_{i-j} \end{aligned} )
$$
$\mathrm{h}_\tau^0 := \mathrm{E}_{\mathrm{s}_\tau}$ 定义为词向量序列。此外，值得一提的是，计算$\mathrm{A}$的一种幼稚的方式是计算所有$(i,j)$对的$\mathrm{W}_{k,R}^n \mathrm{R}_{i-j}$，花费的代价是序列长度的二次方。然而，注意到 $i-j$的值只在0到序列长度之间取值，我们再附录B中给出了一个简单的计算过程，它将代价降低为序列长度的线性。（附录B 参考论文中的附录B吧，论文链接在最后的参考1中）。

### 4 实验

#### 4.1 主要结果

我们将 Transformer-XL 在多种数据集上建立字级别和词级别的语言建模以便与最先进的系统进行比较，包括 WikiText-103(Merity 等，2016），enwik8(LLC,2009)，text8(LLC,2009)，One Billion Word(Chelba 等，2013)和 Penn Treebank(Mikolov 和 Zweig,2012)。

WikiText-103是最大的具有长期依赖性的单词级语言建模基准。它包含来自 28K 篇文章的 103M 训练词汇，平均每篇文章3.6K个词，这可以测试长期依赖模型的能力。我们在训练期间将注意力长度设置为384，在评估期间将注意力长度设置为1600。我们采用自适应 soft-max 和输入表征（Baevski 和 Auli,2018;Grave 等，2016）如表1所示，TransformerXL将之前的state-of-the - art (SoTA) perplexity 从 20.5 减少到 18.3，这说明TransformerXL架构的优越性。

数据集 enwik8 包含 100M 字节未处理的 Wikipedia 文本。我们将我们的架构与表2中以前的结果进行了比较。在模型大小限制下，12层 Transformer-XL 获得了一个新的 SoTA 结果，比 Al-Rfou 等人(2018)的12层 Transformer 高出0.05，而两种 Transformer 的变体都比传统的基于 RNN 的模型有较大的提升。值得注意的是，我们的12层架构实现了与 Al-Rfou 等人(2018)的64层网络相同的结果，只使用了17%的参数预算。为了观察增大模型大小 是否能获得更好的性能，我们训练了18层和24层 Transformer-XLs，训练时的注意长度为784，评估时的注意长度为3800，我们得到了一个新的 SoTA 结果，我们的方法是第一个在广泛研究的字符级基准上突破1.0的方法。与 Al-Rfou 等人(2018)不同，Transformer-XL 不需要任何辅助损失，因此所有的收益都归功于更好的架构。

与enwik8类似但不同的是，text8 包含 100M 经过处理的 Wikipedia 字符，这些字符是通过小写化文本并删除除26个a到z的字母和空格之外的任何字符创建的。由于相似之处，我们简单地将 enwik8 上的最佳模型和相同的超参数应用在 text8 上，而无需进一步调整。表3总结了与以往方法的比较。同样，Transformer-XL 以明显的优势实现了新的 SoTA 结果。

One Billion Word 并没有保留任何长期的依赖关系，因为句子已经被打乱了。因此，该数据集主要测试短期依赖建模的能力。Transformer-XL 与其他方法的比较见表4。虽然 Transformer-XL 的设计主要是为了更好地捕获长期依赖，但它显著地将单模型 SoTA 从 23.7 提高到了 21.8。具体来说，Transformer-XL 明显优于使用普通 Transformer 的当代方法(Baevski和Auli, 2018)，表明Transformer-XL的优势可以推广到短序列建模。

![图3](E:\python_work_space\NLP_Paper_Translation\Transformer-XL_Translation\pic\fig3.png)

我们也在表5中报告了word-level 的penn treebank的结果。与 AWD-LSTM 类似(Merity 等，2017)，我们将变分 dropout 和权重平均值应用于Transformer-XL。通过适当的正则化，Transformer-XL 在没有两步微调的情况下实现了模型中新的 SoTA 结果。Penn Treebank 仅有 1M 训练 tokens,这意味着Transformer-XL 即使在小数据集上也能很好地推广。

![图4](E:\python_work_space\NLP_Paper_Translation\Transformer-XL_Translation\pic\fig4.png)

#### 4.2 消融研究

我们进行了两组消融研究，以检验在 Transformer-XL 中使用的两种技术的效果:递归机制和新的位置编码方案。

第一个研究是在 WikiText-103 上进行的，它需要建模长期依赖关系。结果见表 6。在比较的编码方案中，Shaw 等(2018)是相对的，而 Vaswani 等(2017)和 Al-Rfou 等(2018)是绝对的。“Full”损失和“half”损失是指将交叉熵损失应用于段中的所有或最近的一半位置。我们发现绝对编码只有在一半损失的情况下才能很好地工作，因为一半损失排除了训练中注意力很短的位置，以便更好地进行泛化。表 6显示，为了获得最佳性能，需要使用递归机制和我们的编码方案，并在评估期间将其推广到更长的注意序列。虽然在训练过程中反向传播的长度只有128，但是使用这两种技术，在测试时注意长度可以增加到640。在参数为 151M 的标准设置中，随着注意长度的增加，perplexity 减小。

![image-20210226234853529](E:\python_work_space\NLP_Paper_Translation\Transformer-XL_Translation\pic\fig5.png)

由于递归机制需要额外的内存，我们还将 Transformer-XL 与相同GPU内存约束下的基线进行了比较。如附录A中的表10所示(不准备翻译附录，直接把表10放到上方了)，尽管使用了更短的反向传播长度，Transformer-XL 仍然优于基线。

第二项研究的目标是将解决上下文碎片问题的效果与获取更长的上下文长度的好处分离开来。为了实现这一目标，我们特意选择了一个不需要长期依赖关系的数据集，这样从建立递归得到的任何改进都可以归因于解决上下文碎片问题。具体来说，我们在 One Billion Word 数据集上执行这个受控实验，它只能从去除上下文碎片中获益。我们训练一个20层的 Transformer-XL，其中参数大约为 0.3B，训练 400K 步。如表7所示，即使不需要长期依赖，使用分段级递归也可以显著提高性能，这与我们之前讨论的递归机制解决上下文碎片问题是一致的。此外，我们的相对位置编码在短序列上也优于Shaw等人(2018)。

![image-20210226235021869](C:\Users\xudong\AppData\Roaming\Typora\typora-user-images\image-20210226235021869.png)





#### 4.3 相对有效的上下文长度

Khandelwal等(2018)提出了一种评估序列模型有效上下文长度(Effective Context Length ECL)的方法。ECL是增加上下文跨度将导致增益超过一个阈值的最长长度。然而，ECL忽略了这样一个事实，即当一个模型仅使用较短的上下文就可以获得较低的 perplexity 时，它就更难得到改进，因此不适合在多个模型之间进行公平的比较。相反，我们提出了一个新的度量方法，称为相对有效上下文长度(Relative Effective Context Length, RECL)。RECL是在模型组上而不是单个模型上定义的，长上下文的收益是通过对最佳短上下文模型的相对改进来衡量的。因此，模型组共享相同的基线，以支持公平的比较。RECL还有一个参数$r$，这意味着限制对top-$r$硬示例的比较。有关RECL的更多细节，请参见Appedix C。如表8所示，Transformer-XL 平均使用$r = 0.1$对 900 个单词的依赖关系进行建模。TransformerXL 的 RECL 分别比循环网络和 Transformer 长80%和450%。递归机制和我们的位置编码都有助于更长的 RECL。这进一步证实了我们的论点:Transformer-XL能够对长期依赖进行建模。

#### 4.4 文本生成

只在中等大小的 WikiText-103 上进行了训练，Transforme-XL 已经能够生成具有数千个 tokens 的相对连贯的文章，而无需手动调整（cherry picking），尽管存在一些小缺陷。样品请参考附录E。（参考论文中的附录）

#### 4.5 速度评估

最后，我们将模型的评估速度与通用 Transformer 模型进行比较(Al-Rfou 等， 2018)。如表9所示，由于采用了状态重用方案，Transformer-XL 在评估过程中实现了高达1874倍的加速。

### 5 结论

Transformer-XL 获得了较强的 perplexity 结果，与 RNNs 和 Transformer 相比，它具有更长程的依赖，在评估过程中获得了较大的加速，能够生成连贯的文本文章。我们展望了 Transformer-XL 在文本生成、无监督特征学习、图像和语音建模等领域的有趣应用。



### 参考文献



### 附录

附录部分涉及到实验结果的对比、高效的计算相对位置编码公式推导、RECL的相关细节、可视化Attention。另外作者还花了几页篇幅展示了文本生成的实验结果。有兴趣的请参考原论文，这里不再翻译。

