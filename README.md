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


