

# Recommend Paper Reproduction with PyTorch and Tensorflow 

<p align="left">
  <img src='https://img.shields.io/badge/python-3.7-blue'>
  <img src='https://img.shields.io/badge/pytorch-1.14-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-2.0-blue'>
</p>  

最近从CV转行做推荐，阅读了许多用深度学习做推荐的paper，有感而发，觉得推荐领域论文的工程性大都很强，很多都是从实际的业务和数据出发，在解决业务需求的同时能产出一篇不错的paper。同时在自己学习的过程中觉得paper上读到的东西终究有限，需要自己实现一下论文方法，才能有更加深刻的理解。


##Introduction：
- 本项目旨在对自己读过的和感兴趣的深度学习推荐系统领域的文章进行复现
- 提供**PyTorch**和**Tensorflow2.x**两种框架的实现
- 使用**lmdb数据库**加速数据加载过程
- 支持单GPU和多GPU训练加速


##Dataset：
- [Criteo](./dataset/Criteo.md)
- Taobao ctr[TODO]

##Some importent details
###Regularization:
为了减少过拟合风险，为模型添加l2正则，PyTorch和TensorFlow两者的实现略有不同

* **PyTorch**:可以指定torch.nn.optim.Adam的weight_decay参数实现l2正则，如果只希望对部分参数（如weight）正则化可使用如下方式：

``` python
# pytorch使用weight_decay添加L2正则化（对bias 和 bn不进行l2正则）
    weight_decay_list = (param for name, param in model.named_parameters()
                         if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters()
                     if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.0}]
    optimizer = optim.Adam(parameters, lr=0.001, weight_decay=5e-4)
```

* **TensorFlow**：在定义模型参数的时候就对其添加l2正则

``` python
self.weight_dense = self.add_weight(name='weight_dense',
                                    shape=[int(input_shape[0][-1]), 1],
                                    initializer=glorot_normal(seed),
                                    regularizer=l2(self.l2_reg),
                                    trainable=True)
```

###Parmater initialization
参数初始化的时候我们希望输入和输出的方差相同，使用Xavier初始化方法

* **PyTorch**: 先定义参数大小，再填入相应的初始化值

``` python
 self.weight = nn.ParameterList([nn.Parameter(torch.empty(self.field_dim, 1)) for _ in range(self.cross_num)])
 for weight in self.weight:
 	nn.init.xavier_normal_(weight， gain=1)
```

* **TensorFlow**: 定义参数的时候直接指定相应的初始化

``` python
self.weight_dense = self.add_weight(name='weight_dense',
                                    shape=[int(input_shape[0][-1]), 1],
                                    initializer=glorot_normal(seed),
                                    regularizer=l2(self.l2_reg),
                                    trainable=True)
```

##Usage example
以DeepFM为例，具体的步骤如下：

1. 数据预处理，参见Data process
2. 指定训练参数

``` python
params = [
        '--epochs',      '100', 
        '--batch_size',  '4096',
        '--optimizer',   'Adam',
        '--loss',        'BCE',   # loss func is Binary CrossEntropy Loss
        '--gpu_ids',     '1, 2, 3, 4', # 用4张卡并行训练
        '--lr',          '0.001',
        '--embed_size',  '8',
        '--model_save_dir', './checkpoints/DeepFM/'
    ]
```

3. 训练
	- PyTorch```python ./train_example_pytorch.py```
	- Tensorflow ```python ./train_example_tensorflow.py```


##models

|  Publish | paper |  model |
| :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: |
| ICDM 2010|[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | FM | 
| RecSys 2016|[Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) | FFM |                                                     
| IJCAI 2017|[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)|DeepFM|
| ADKDD 2017|[Deep&Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)| Deep&Cross|
| IJCAI 2017|[Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435)  | AFM |
| KDD 2018|[xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) | xDeepfm |    
| CIKM 2019|[AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)| AutoInt |


##TODO
###model
|  Publish | paper |  model |
| :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: |
| [arxiv 2019]|[Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)|  ONN |
| [RecSys 2019]|[FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf) | FiBiNET |
| [arxiv 2019]|[Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf) | DIN| 
| [AAAI 2019]|[Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf) | DIEN|
| [arxiv 2020]|[DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) |DCN V2|





###Contact
如果对本项目有任何疑问或者建议，请邮件**Sunrise2019@zju.edu.cn**或者添加我的微信号**13735593240**



