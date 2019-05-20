## 运行说明

以Kaggle细胞图为例运行,数据下载链接 https://www.kaggle.com/c/data-science-bowl-2018/data



**1.环境安装**

主要依赖环境包参考 `setup_env.sh`文件



**2.数据处理**

* 把原始数据放进`data`目录,原始数据结构如下:

![数据目录结构](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/%E6%95%B0%E6%8D%AE%E7%9B%AE%E5%BD%95.png)

* 运行`create_stage1_labels.py`为图像生成label文件
* 运行`create_mask.py`为每个图像添加边界,效果如下:

![massk图添加边界](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/mask%E8%BE%B9%E7%95%8C%E5%9B%BE.png)

* 备注:原始数据中会包含一份`fold.csv`文件,用于交叉验证集划分数据,实际情况中如果没有,可以自行通过`from sklearn.model_selection import StratifiedKFold`方法进行交叉验证集划分,修改`albu/srcutils.py`文件中的`get_csv_folds`函数,返回划分的数据索引



**3.训练**

* 修改`albu/src/configs/`里面的配置文件设置,设置选择模型名称,训练参数,数据路径,模型保存位置等,配置文件中`network`可选列表如下:

  * `resnet34_upsample`: unet.Resnet34_upsample
  * `resnet34_sum`: unet.Resnet34_sum
  * `resnet34_double`: unet.Resnet34_double
  * `resnet34_bn`: unet.Resnet_bn_sum
  * `resnet34_dil`: unet.DilatedResnet34
  * `dpn`: unet.DPNunet
  * `incv3`: unet.Incv3
  * `resnet38`: resnet38unet.WideResnet38
  * `vgg11bn`: unet.Vgg11bn
  * `vgg16bn`: unet.Vgg16bn

  配置文件中`optimizer`可选列表如下:

  * `adam`: optim.Adam
  * `rmsprop`: optim.RMSprop
  * `sgd`: optim.SGD

* 执行`train_all.sh`文件进行训练



**4.预测**

* 测试数据放到`data_test`目录下面,目录结构如下:

  ![测试数据目录结构](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/%E6%B5%8B%E8%AF%95%E6%95%B0%E6%8D%AE%E7%9B%AE%E5%BD%95.png)

* 修改`albu/src/configs`里面的配置文件,设置模型加载路径,预测数据路径,预测结果保存路径等

* 执行`predict_test.sh`文件

* 预测结果如下:

![测试1](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/test1.png)

![测试2](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/test2.png)

![测试3](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/test3.png)

![测试4](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/test4.png)

![测试5](https://git.sg-ai.com/hzhou/my_ai/raw/master/DPNunet/test_images/test5.png)



