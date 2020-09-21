import os
import tokenization

# bert 路径定义
bert_parameter_location = './pretrain_weights/chinese_L-12_H-768_A-12/'
# bert解码器引入
bert_encoder = tokenization.BERTTextEncoder(os.path.join(bert_parameter_location, 'vocab.txt'))
# 项目名
projectName = 'relation_extraction_bert_集中注意力语言模型_联合命名实体识别任务'

# NER 表
neid2type = {
    0: 'null',
    1: 'O',
    2: 'B-body',
    3: 'I-body',
    4: 'E-body',
    5: 'B-grade',
    6: 'E-grade',
    7: 'B-negative',
    8: 'E-negative',
    9: 'I-grade',
    10: 'B-location',
    11: 'I-location',
    12: 'E-location',
    13: 'B-num',
    14: 'I-num',
    15: 'E-num',
    16: 'S-negative',
    17: 'S-location',
    18: 'S-body',
    19: 'S-grade',
    20: 'S-num'
}

# RE 关系表
# 0: 'null', （个人习惯）
# 1: 'no_relation',
reid2type = {
    0: 'null',
    1: 'no_relation',
    2: 'grade_of_1_2',
    3: 'grade_of_2_1',
    4: 'locate_1_2',
    5: 'locate_2_1',
    6: 'neg_of_2_1',
    7: 'num_of_1_2',
    8: 'num_of_2_1'
}


netype2id = {v:k for k,v in neid2type.items()}
retype2id = {v:k for k,v in reid2type.items()}