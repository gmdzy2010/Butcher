# `屠夫`项目
注：`屠夫`是游戏`暗黑1/暗黑3`里面第一幕的Boss，希望跨过屠夫能够正式入门深度学习领域

`深度学习`是`机器学习`的子集
* 机器学习：将`输入`映射到`目标`，这一过程是通过观察许多输入和目标的示例来完成的
* 深度学习：将`输入`映射到`目标`，这一过程是通过深度神经网络的数据变换来完成的

### CNN(卷积神经网络)
常用于图像分类（图像模式识别）
1. 底层堆叠数个由`卷积神经网络层`和`池化层`，负责从输入提取`特征图`
2. 获取特征图后使用`展平层`连接
3. 最后使用数个`全连接层`作为分类器，完成对输入的分类

##### 特征提取
TODO

##### 模型微调
TODO

### RNN(循环神经网络)
常用于文本分类

TODO

### LSTM(长短期记忆)
常用于稳定时间序列的预测

TODO

### GAN(对抗生成网络)
常用于文本、图像生成任务

TODO

## 项目结构
```
.
├── README.md                     项目介绍
├── apps                          项目总目录
│   ├── __init__.py
│   ├── image_classification      图像分类
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── datasets.py           数据集
│   │   └── models.py             训练的模型
│   ├── temprature_prediction     温度预测
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   └── rnn.py
│   ├── text_classification       文本分类
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   └── rnn.py
│   └── text_generation           文本生成
│       ├── __init__.py
│       ├── datasets.py
│       ├── models.py
│       └── rnn.py
├── data                          输入、输出数据目录
│   ├── images                    训练、验证和测试用的图片
│   ├── models                    保存的模型文件.keras
│   └── texts                     训练、验证和测试用的文本
├── deploy
│   ├── prod.Dockerfile           build镜像使用
│   └── requirements.txt          poetry 生成，生产依赖
├── main.py                       应用入口程序
├── poetry.lock                   poetry 生成
├── poetry.toml                   poetry 本地配置
├── pyproject.toml                项目信息、开发环境配置
└── utils                         应用用到的组件
    ├── __init__.py
    ├── exporter.py               模型导出等功能
    └── image_plotter.py          使用模型的训练数据绘图
```

## 项目运行
本项目使用`Python 3.12.2`，`3.12.0`版本经测试不能保存模型文件
```sh
python main.py
```
