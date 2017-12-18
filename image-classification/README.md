# Image Classification with ava advancd sdk

本文介绍在通用的图片分类训练中如何使用ava 高级训练接口管理训练过程。

## Contents

1. [ava高级训练SDK](#ava高级训练SDK)

## ava高级训练SDK

参考 [ava 高级训练SDK文档](https://github.com/qbox/ava/blob/dev/docs/Ava.sdk.md) 中的[例子](https://github.com/qbox/ava/blob/dev/docker/scripts/examples/mxnet/training.py)

python ava SDK 目前支持的 MxNet 分类中的功能包括：
- 启动一个训练实例，此时会在ava portal 留下一条训练的记录
- 处理中断信号，判断训练是否正常结束，会在ava portal上改变训练记录的状态
- 定义了回调函数，替换原有回调函数

使用姿势：
- [新建一个training_instance](https://github.com/likelyzhao/training_with_ava/blob/master/image-classification/train_mnist_ava.py#L87)
- [注册消息处理函数](https://github.com/likelyzhao/training_with_ava/blob/master/image-classification/train_mnist_ava.py#L50-L58)
- [正常训练分类](https://github.com/likelyzhao/training_with_ava/blob/master/image-classification/train_mnist_ava.py#L50-L58)
- [添加中断处理函数](https://github.com/likelyzhao/training_with_ava/blob/master/image-classification/train_mnist_ava.py#L106-L109)

