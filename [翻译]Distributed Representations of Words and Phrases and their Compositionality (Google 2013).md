# Distributed Representations of Words and Phrases and their Compositionality 



最近引入的连续Skip-gram模型是学习高质量的分布式向量表示（distributed vector representations）的有效方法，分布式向量表示可以捕获大量精确的句法和语义关系。在本文中，我们提出了几个扩展，提高向量的质量和训练速度。通过对频繁词(frequent words)的二次采样，我们获得了明显的加速，同时还学习到了更多的规律的词表示(regular word representations)。我们还提出了一个分层 softmax 的简单替代方案，称为negative sampling（负采样）。

word representations的一个固有限制是：它们不关心词序，而且无法表示惯用语。例如，它不能简单的将 Canada 和 Air 的含义组合起来得到 Air Canada(加拿大航空公司) 。在这个例子的启发下，我们提出了一种在文本中查找短语的简单方法，并表明学习数百万个短语的向量表示是可能的。

