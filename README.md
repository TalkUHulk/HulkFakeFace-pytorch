
# HulkFaceZoo-pytorch
![TalkUHulk](https://img.shields.io/badge/TalkUHulk-Python3-green)  
## 简介

HulkFaceZoo为个人的人脸相关项目实践学习库，主要包括检测（retinaface）+识别（arcface）+相关应用（如颜值打分、检索），相关代码和预训练模型都可在本项目中找到。

🔥水平有限，有错误请及时指出，共同进步学习～

## 相关模型及数据集
链接: https://pan.baidu.com/s/1WDg6NwQODyll1Qour5GeAQ  密码: tw16

## 环境
- python3
- torch==1.6.0 
- torchvision==0.7.0 
- opencv_contrib_python==4.4.0.44
- warmup_scheduler==0.3.2
- tensorboardX==2.1
- pymilvus==0.2.14

 ## RetinaFace
 [参考项目](https://github.com/biubug6/Pytorch_Retinaface)

 - **backbone**: *mobilenet-v2 ｜ resnet50*
 - **datasets**: *widerface*
 - **addition**: *在原始网络中加入性别判断，目前仅在resnet50中支持，*[cfg_re50.gender=True](https://github.com/TalkUHulk/HulkFakeFace-pytorch/blob/master/RetinaFace/data/config.py)
 - **tensorboard**: 
 
  ![logs](https://github.com/TalkUHulk/HulkFakeFace-pytorch/blob/master/logs/RetinaFace.jpg)
  
 - **FDDB ROC**:
 
  ![ROC](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/RetinaFace/evalution/ROC.png)
  
 - **demo**:
 
  ![girls](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/RetinaFace/results/girl.jpg)
  
  ![nba](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/RetinaFace/results/nba.jpg)
  
  ![stars](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/RetinaFace/results/stars.jpg)
  
 ## GenderClassify
 - **简介**: *主要用来标注widerface，为RetinaFace with gender制作训练集*
 
 ## ArcFace 
 
 - **Pretrained Models**: *在[InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) 的基础上，使用亚洲人脸fine-tune*
 - **Performance**:  
 
 | LFW(%) |  
 | ------ |  
 | 0.9943 |  
 - **tensorboard**:   
 ![Loss](https://github.com/TalkUHulk/HulkFakeFace-pytorch/blob/master/logs/ArcFace_Loss.jpg)
 
 ![Arc](https://github.com/TalkUHulk/HulkFakeFace-pytorch/blob/master/logs/ArcFace_Acc.jpg)
 
 ## Beauty
  - **loss**: train with arcface and pretrained with MarginRankingLoss
  - **datasets**: *SCUT_FBP5500*
  - **Performance**:   
  
 |SCUT_FBP5500-MAE|  
 | ------ |  
 | 0.277 |  
  - **tensorboard**:   
   ![Train](https://github.com/TalkUHulk/HulkFakeFace-pytorch/blob/master/logs/Beauty_train.jpg)  
   
 ![Val](https://github.com/TalkUHulk/HulkFakeFace-pytorch/blob/master/logs/Beauty_val.jpg)

 
 ## Retrieval 
 
```
Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.
```
 - **datasets**[明星人脸](http://www.seeprettyface.com/mydataset_page3.html#star)
 - **milvus**: [向量搜索引擎](https://www.milvus.io/cn/)
 - **demo**:   
 
 ![ab](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/FaceRetrieval/result/result_ab.jpg)  
 
 ![st](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/FaceRetrieval/result/result_st.jpg)
 
 ![tly](https://github.com/TalkUHulk/HulkFakeFace-pytorch/tree/master/FaceRetrieval/result/result_tongliya.jpg)


## References 

 - [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
 
 - [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
 