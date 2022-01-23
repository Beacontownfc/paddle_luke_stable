# LUKE_paddle_stable

## 1 简介 

**本项目基于是先前复现的LUKE的稳定版本，按照要求复现改进如下:**

- 在open entity数据集上成功达到论文精度
- 更稳定的运行结果
- 在SQuAD1.1数据集上我们提供多卡运行版本
- LukeTokenizer我们重新进行复现，无需依赖transformers库
- 我们提供aistudio notebook, 帮助您快速验证模型

**项目参考：**
- [https://github.com/studio-ousia/luke](https://github.com/studio-ousia/luke)

**原复现地址：**
- [https://github.com/Beacontownfc/paddle_luke](https://github.com/Beacontownfc/paddle_luke)

## 2 复现精度
>#### 在Open Entity数据集的测试效果如下表。
>在open entity数据集上我们成功达到论文精度，最高精度比原论文高出0.3%(所有超参与原论文代码一致)

|网络 |opt|batch_size|数据集|F1|F1(原论文)|
| :---: | :---: | :---: | :---: | :---: | :---: |
|Luke-large|AdamW|2|Open Entity|78.54|78.2|

>复现代码训练日志：
[复现代码训练日志](open_entity/train.log)
>
>#### 在SQuAD1.1数据集的测试效果如下表。
>由于SQuAD1.1数据集比较特殊，不提供测试集，因此对比验证集的结果
>
>在SQuAD1.1数据集上，成功复现了论文精度

|网络 |opt|batch_size|数据集|F1|F1(原论文)|EM|EM(原论文)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
|Luke-large|AdamW|8|SQuAD1.1|94.95|95.0|89.73|89.8

>复现代码及训练日志：
[复现代码训练日志](reading_comprehension/train.log)
>
## 3 数据集
下载Open Entity数据集
[下载地址](https://cloud.tsinghua.edu.cn/f/6ec98dbd931b4da9a7f0/)
把下载好的文件解压,并把解压后的Open Entity目录下的`train.json`、`test.json`和`dev.json`分别为训练集、验证集和测试集或者可以直接使用`./open_entity/data`路径下的open entity数据集

下载SQuAD1.1数据集，主流机器阅读理解数据集
[下载地址](https://data.deepai.org/squad1.1.zip)

同时需要下载由LUKE官方提供的维基百科(实体)数据集
[下载地址](https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view)

## 4环境依赖
运行以下命令即可配置环境
```bash
pip install -r requirements.txt
```

## 5 快速开始
如果你觉得以下步骤过于繁琐，您可以直接到此处
[链接](https://aistudio.baidu.com/aistudio/projectdetail/3393133)
快速验证open entity数据集上的结果，以及此处
[链接](https://aistudio.baidu.com/aistudio/projectdetail/3438351)
快速验证SQuAD1.1数据集上的结果，以上均在AIStudio Notebook上运行。
#### 数据集下载好后，同时下载预训练权重: [下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/123707)

###### 训练并测试在open entity数据集上的F1：
###### 进入到`./open_entity`文件夹下, 运行下列命令


```bash
python main.py --do_train=1 --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR> --checkpoint_file=<NAME> --pretrain_model=<MODEL>
```

评估训练好的模型:

```bash
python main.py --do_eval=1 --data_dir=<DATA_DIR> --output_dir=<OUTPUT_DIR> --checkpoint_file=<NAME> --pretrain_model=<MODEL>
```

说明：

- `<DATA_DIR>`、`<MODEL>`和`<OUTPUT_DIR>`分别为数据集文件夹路径、预训练权重路径和输出文件夹路径, `<NAME>为你自定checkpoint名字`

- 若想要得到论文精度，需要多运行几次，运行一次在v100 16GB上大概15分钟

运行结束后你将看到如下结果:
```bash
Results: %s {
  "test_f1": 0.7815726767275616,
  "test_precision": 0.7880405766150561,
  "test_recall": 0.7752100840336135
}
```

###### 训练并测试在SQuAD1.1数据集上的F1：
###### 进入到`./reading_comprehension`文件夹下, 运行下列命令

首先预处理数据集：
```bash
python create_squad_data.py --wiki_data=<WIKI_DATA_DIR> --data_dir=<DATA_DIR1> --output_data_dir=<DATA_DIR2>
```
运行结束后你将看到预处理好数据的json和pickle文件：`train.json`、`eval_data.json`和`eval_obj.pickle`，存放在`<DATA_DIR2>`路径下

```bash
python -m paddle.distributed.launch main.py --data_dir=<DATA_DIR2> --pretrain_model=<MODEL> --output_dir=<OUTPUT_DIR> --multi_cards=1 --do_train=1
```

以上为多卡训练，使用单卡训练如下:
```bash
python main.py --do_train=1 --data_dir=<DATA_DIR2> --checkpoint_file=<NAME> --output_dir=<OUTPUT_DIR> --pretrain_model=<MODEL>
```
说明： 

- `<WIKI_DATA_DIR>`为LUKE官方提供的维基百科(实体)数据集文件夹路径, `<DATA_DIR1>`为SQuAD1.1数据集路径, `<DATA_DIR2>`解释如上，`<NAME>`是你自定checkpoint名字, `<MODEL>`是预训练权重路径

- 若想要得到论文精度，需要多运行几次，运行一次在四张v100 32GB上大概75分钟

评估训练好的模型:
```bash
python main.py --do_eval=1 --data_dir=<DATA_DIR1> --checkpoint_file=<NAME> --output_dir=<OUTPUT_DIR>
```

运行结束后你将看到如下结果:
```bash
{"exact_match": 89.73509933774834, "f1": 94.95971612635493}
```



## 6 代码结构与详细说明
```
├─open_entity
| ├─data                     # 数据集文件夹
| | ├─merges.txt             #tokenizer 文件
| | ├─entity_vocab.tsv       #实体词文件
| | ├─vocab.json             #tokenizer 文件
| ├─luke_model               #LUKE模型文件
| | ├─utils
| | ├─entity_vocab.py
| | ├─interwiki_db.py
| | ├─model.py   
| ├─datagenerator.py         #数据生成器文件
| ├─main.py                  #运行训练并测试
| ├─open_entity.py           #LUKE下游任务
| ├─trainer.py               #训练文件
| ├─utils.py                   
├─reading_comprehension        
| ├─luke_model
| | ├─utils
| | ├─model.py
| ├─squad_data
| | ├─entity_vocab.tsv
| | ├─merges.txt
| | ├─metadata.json
| | ├─vocab.json
| ├─src
| ├─utils
| ├─create_squad_data.py
| ├─main.py
| ├─reading_comprehension.py         #LUKE下游任务                                       
```