# FlyAI_ButterflyClassification
FlyAI的一个分类竞赛
## 说明
- `main.py` 用于训练一键训练，不需要任何参数。如果想修改参数, 在这个路径 `./configs/xxx.yaml` 中修改。
- 3种模型经过后处理分别预测，最后分数融合。
- 如果在FlyAi中跑，则需要把文件全部拷贝到一个文件夹中，然后压缩上传。不拷文件，直接下载这个仓库, 会多一个master文件夹，则文件结构如下，会报错。
```
|------master---|
                |---config
                |---configs
                |---libs
                ...
```
把文件全部拷贝到一个文件夹中, 结构如下，不会报错。
```
|---config
|---configs
|---libs
...
```
## 模型支持
- [x] ResNet
- [x] SE_ResNet
- [x] SE_ResNext
- [x] ResNet_ibn
- [x] EfficientNet
- [x] ResNest
## 损失函数支持
- [x] centerloss
- [x] softmaxloss
## 优化器支持
- [x] Ranger
- [x] SGD
## 学习率调整策略
- [x] multistep
- [x] warmup
## 预处理支持
- [x] mixup
- [x] AutoAug
- [x] RandomErasing
- [x] GridMask
## 后处理支持
- [x] TestTimeAugmentation, [TTA](https://github.com/BloodAxe/pytorch-toolbelt)

