# pytorch_project

# 1. project description
pytorch model and some paper notes

# 2. contents
## 2.1 Xchest picture recognization
肺部感染图像识别，模型微调。


## 2.2 GRU4Rec model
- 一般的RNN模型我们的输入和输出是什么，我们对RNN输入一个序列 $X = [x^1,x^2,...,x^n]$ ，注意我们序列中的每一个节点都是一个向量，那么我们的RNN会给我们的输出也是一个序列 $Y = [y^1,y^2,...,y^n]$, 那我们如何通过RNN提取出输入序列的特征呢，常用做法有两种（GRU4Rec用的第一种获得序列表征）：
   - 取出 $y^n$ 的向量表征作为序列的特征，这里可以认为 $x^1, x^2, \ldots, x^n$ 的所有信息，所有可以简单的认为$y^n$的结果代表序列的表征
   - 对所有时间步的特征输出做一个Mean Pooling，也就是对 $Y = [y^1,y^2,...,y^n]$ 做均值处理，以此得到序列的表征
- GRU4Rec亮点在于设计了mini-batch和高效的sampling方法。
   - 本质是多分类问题：得到用户embedding后，直接通过多分类进行损失计算，多分类的标签是用户下一次点击的item的index，直接通过User的向量表征和所有的Item的向量做内积算出User对所有Item的点击概率（一个`score`），然后通过`Softmax`进行多分类损失计算 。
   - listwise训练模式的Loss 为 `torch.nn.CrossEntropyLoss`, 即对输出进行 Softmax 处理后取交叉熵。
   - 本task还学习`faiss`向量检索库和`TSNE`的Embedding分布可视化。
- 回顾DIEN：利用 AUGRU(GRU with Attention Update Gate) 组成的序列模型，**在兴趣抽取层GRU基础上加入注意力机制，模拟与当前目标广告（Target Ad）相关的兴趣进化过程**，兴趣进化层的最后一个状态的输出就是用户当前的兴趣向量 h'(T)。 


## 2.3 MIND
- Hinton在2011年提出的capsule network，通过EM期望值最大化算法，用动态路由代替反向传播，学习不同capsule之间的连接权重，实现比CNN更优秀的空间关系建模效果（CNN可能对同一个图像的旋转版本识别错误）。
- 阿里的序列召回MIND模型：引入了胶囊网络，==将胶囊网络的动态路由算法引入到了用户的多兴趣建模上==，通过B2I动态路由很好的从用户的原始行为序列中提取出用户的多种兴趣表征，其实就是将用户行为聚类成多个类表示用户兴趣。
- 在离线训练阶段，通过提出Label-Aware Attention详细的探讨了多兴趣建模下的模型参数更新范式。
- 序列召回推荐本质是一个多分类问题，将用户历史`item_emb`进行pooling后的`user_emb`，和`item_emb`进行内积得到`score`偏好分数。交叉熵损失函数的参数即`socre`和标签`item_id`（该用户交互的`item_id`），典型的多分类问题。

## 2.4 Comirec-DR
- 多兴趣召回建模。Comirec论文中的提出的第一个模型：Comirec-DR（DR就是dynamic routing），阿里将用户行为序列的item embeddings作为初始的capsule，然后提取出多个兴趣capsules，即为用户的多个兴趣。其中胶囊网络中的动态路由算法和MIND类似，不同在于：
   - 输入序列胶囊$i$与所产生的兴趣胶囊 $j$ 的权重 $b_{i j}$初始化为0；
   - 在Comirec-DR中对于不同的序列胶囊i与兴趣胶囊j，我们都有==一个独立的$W_{i j} \in \mathbb{R}^{d \times d}$来完成序列胶囊`i`到兴趣胶囊`j`之间的映射==。
   
   
## 2.5 Comirec-SA
- Comirec-SA基于attention的多兴趣建模，论文中先通过attention提取单一兴趣，再推广到多兴趣建模。另外使用贪心算法优化带有准确度+多样性的目标函数。
- DR把MIND的attention换成argmax（还有初始化方式不同、序列胶囊到兴趣胶囊用可学习权重矩阵替代的不同），SR则本质是多头注意力进行多兴趣建模。
- `torch.einsum`（Einstein summation convention，即爱因斯坦求和约定）可以写tensor运算和更高维度tensor写法更加简洁，如用`torch.einsum("bij, bjk -> bik", batch_tensor_1, batch_tensor_2)`进行batch内矩阵乘法运算。
