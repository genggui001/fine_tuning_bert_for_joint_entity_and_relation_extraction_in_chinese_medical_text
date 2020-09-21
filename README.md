# Fine-tuning BERT for joint entity and relation extraction in chinese medical text

## 用法

### 下载预训练参数
从<a href="https://github.com/google-research/bert">google-research/bert</a>下载<a href="https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip">BERT-Base, Chinese</a>解压并放入pretrain_weights文件夹下

### 准备数据集
将关系抽取数据集转换成如下格式
```
[
    {
        "text": "左前降支近段管壁不规则，最狭窄处30-40％。",
        "entities": [
            [
                0,
                4,
                "左前降支",
                "body"
            ],
            [
                4,
                6,
                "近段",
                "location"
            ],
            [
                13,
                15,
                "狭窄",
                "grade"
            ],
            [
                16,
                22,
                "30-40％",
                "num"
            ]
        ],
        "relations": [
            [
                0,
                1,
                "locate_1_2"
            ],
            [
                0,
                2,
                "grade_of_2_1"
            ],
            [
                2,
                3,
                "num_of_1_2"
            ]
        ]
    },
    ...
]
```

### 再将数据集分割保存到

- data_set/train.json
- data_set/dev.json
- data_set/test.json

### 修改config.py中的neid2type以及reid2type
```
# NER 表
neid2type = {
    0: 'null',
    1: 'O',
    2: 'B-xxxx',
    3: 'I-xxx',
    4: 'E-xxxx',
    5: 'S-xxxx',
    .......
}

# RE 关系表
# 0: 'null', （个人习惯）
# 1: 'no_relation',
reid2type = {
    0: 'null',
    1: 'no_relation',
    2: 'xxxxx_of_1_2',
    3: 'xxxxx_of_2_1',
    .........
}
```

### 安装conda依赖
```
conda env create -f environment.yml
```

### 激活环境并运行
```
conda activate tf1.12
python main.py
```

## 引用
- https://arxiv.org/abs/1908.07721
```
@inproceedings{xue2019fine,
  title={Fine-tuning BERT for joint entity and relation extraction in chinese medical text},
  author={Xue, Kui and Zhou, Yangming and Ma, Zhiyuan and Ruan, Tong and Zhang, Huanhuan and He, Ping},
  booktitle={2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={892--897},
  year={2019},
  organization={IEEE}
}
```

## 鸣谢
参考了以下代码

<a href="https://github.com/google-research/bert">google-research/bert</a>

<a href="https://github.com/Separius/BERT-keras">Separius/BERT-keras</a>