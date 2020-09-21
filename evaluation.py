from seqeval.metrics.sequence_labeling import get_entities
from data.dataset import generate_pos_ids
from config import *

import numpy as np


def eval_prf(data_predict, data_true):
    type_map = {}
    
    for items_predict, items_true in zip(data_predict, data_true):
        tmp_type_set_map = {}
        
        for item_predict in items_predict:
            if item_predict[-1] not in tmp_type_set_map:
                tmp_type_set_map[item_predict[-1]] = [set(), set()]
            tmp_type_set_map[item_predict[-1]][0].add(item_predict)
        
        
        for item_true in items_true:
            if item_true[-1] not in tmp_type_set_map:
                tmp_type_set_map[item_true[-1]] = [set(), set()]
                
            tmp_type_set_map[item_true[-1]][1].add(item_true)

        for key, value in tmp_type_set_map.items():
            if key not in type_map:
                type_map[key] = {'p': 0.0, 'r': 0.0, 'f1': 0.0, 'pred_count': 0, 'true_count': 0, 'right_count': 0}
                
            type_map[key]['pred_count'] += len(value[0])
            type_map[key]['true_count'] += len(value[1])
            type_map[key]['right_count'] += len(value[0] & value[1])
        
    
    ALL = {'p': 0.0, 'r': 0.0, 'f1': 0.0, 'pred_count': 0, 'true_count': 0, 'right_count': 0}
    
    for key, value in type_map.items():
        ALL['pred_count'] += value['pred_count']
        ALL['true_count'] += value['true_count']
        ALL['right_count'] += value['right_count']
    
    type_map['ALL'] = ALL
    
    for key, value in type_map.items():
        value['p'] = 0 if value['pred_count'] == 0 else value['right_count'] / value['pred_count']
        value['r'] = 0 if value['true_count'] == 0 else value['right_count'] / value['true_count']
        value['f1'] = 0 if value['p'] + value['r'] == 0 else 2 * value['p'] * value['r'] / (value['p'] + value['r'])
        
        print("%s:\tPrecision:%.2f\tRecall:%.2f\tF-score:%.2f\tlenRight:%d\tlenPred:%d\tlenVal:%d" % (key,value['p']*100, value['r']*100, value['f1']*100, value['right_count'], value['pred_count'], value['true_count']))
    
    
    return type_map['ALL']['f1']

def eval_model(model_data, model, max_len):
    #真确答案保存
    mapped_ner_true = []
    mapped_re_true = []
    
    # 先计算实体 ，构造实体预测输入

    input_ids = []
    input_masks = []
    input_type_ids = np.zeros((len(model_data), max_len), dtype=np.int32)
    
    input_masks = []
    input_entity_masks = []

    for item in model_data:
        text = item['text']
        entities = item['entities']
        relations = item['relations']

        tokens, char_to_word_offset =  bert_encoder.tokenize(text)

        maped_entities = []

        for start_pos, end_pos, _, entity_type in entities:

            map_start_pos = start_pos
            while char_to_word_offset[map_start_pos] is None:
                map_start_pos += 1
            map_start_pos = char_to_word_offset[map_start_pos]

            map_end_pos = end_pos - 1
            while char_to_word_offset[map_end_pos] is None:
                map_end_pos += 1
            map_end_pos = char_to_word_offset[map_end_pos] + 1

            maped_entities.append((map_start_pos, map_end_pos, entity_type))
        
        mapped_ner_true.append(maped_entities)
        mapped_re_true.append([tuple(list(maped_entities[start_entity_idx]) + list(maped_entities[end_entity_idx]) + [relation_type]) for start_entity_idx, end_entity_idx, relation_type in relations])
        
        input_tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        
        input_id = bert_encoder.standardize_ids(bert_encoder.convert_tokens_to_ids(input_tokens))
        input_mask = [1] * len(input_tokens)
        input_entity_mask = [0] * len(input_id)
    
        input_id += [0] * (max_len - len(input_id))
        input_mask += [0] * (max_len - len(input_mask))
        input_entity_mask += [0] * (max_len - len(input_entity_mask))
        
        
        input_ids.append(input_id)
        input_masks.append(input_mask)
        input_entity_masks.append(input_entity_mask)
    
    input_ids=np.array(input_ids, dtype=np.int32)
    input_type_ids=np.array(input_type_ids, dtype=np.int32)

    
    input_masks=np.array(input_masks, dtype=np.int32)
    input_entity_masks=np.array(input_entity_masks, dtype=np.int32)
    
    pos = generate_pos_ids(len(model_data), max_len)
    
    x = [input_ids, input_type_ids, pos, input_masks, input_entity_masks]
    
    #预测实体
    y_predict = model.predict(x, batch_size=128, verbose=1)

    ne_predict = []
    
    for item in np.argmax(y_predict[1], axis=-1):
        BIOES_list = []
        
        #删除头部CLS
        for item in item[1:]:
            if item == 0:
                BIOES_list.append("O")
            else:
                BIOES_list.append(neid2type[item])
        
        
        ne_predict.append([(start, end + 1, entity_type) for entity_type, start, end in get_entities(BIOES_list)])
    
    # 计算正确率
    ne_f1 = eval_prf(ne_predict, mapped_ner_true)
    
    # 预测re
    re_predict_map = {}
    
    # 输入构造
    input_ids = []
    input_masks = []
    input_type_ids = []
    
    input_masks = []
    input_entity_masks = []
    
    for data_idx, item in enumerate(model_data):
        text = item['text']
        tokens, char_to_word_offset =  bert_encoder.tokenize(text)
        
        maped_entities = ne_predict[data_idx]
        re_index = 0
        for start_entity_idx in range(len(maped_entities)):
            for end_entity_idx in range(start_entity_idx + 1, len(maped_entities)):
#                 remask = [0 for _ in range(len(tokens))]
                
#                 for tmp_pos in range(maped_entities[start_entity_idx][0], maped_entities[start_entity_idx][1]):
#                     remask[tmp_pos] = 1
                
#                 for tmp_pos in range(maped_entities[end_entity_idx][0], maped_entities[end_entity_idx][1]):
#                     remask[tmp_pos] = 1
                
#                 remask_list.append(remask)
                
                input_tokens = ['[CLS]'] + tokens + ['[SEP]'] 

                input_id = bert_encoder.standardize_ids(bert_encoder.convert_tokens_to_ids(input_tokens))
                input_mask = [1] * len(input_tokens)
                input_entity_mask = [0] * len(input_id)
                
                input_entity_mask[0] = 1
                
                for tmp_pos in range(maped_entities[start_entity_idx][0], maped_entities[start_entity_idx][1]):
                    input_entity_mask[tmp_pos + 1] = 1

                for tmp_pos in range(maped_entities[end_entity_idx][0], maped_entities[end_entity_idx][1]):
                    input_entity_mask[tmp_pos + 1] = 1
                

                input_id += [0] * (max_len - len(input_id))
                input_mask += [0] * (max_len - len(input_mask))
                input_entity_mask += [0] * (max_len - len(input_entity_mask))
                
                input_ids.append(input_id)
                input_masks.append(input_mask)
                input_entity_masks.append(input_entity_mask)
                
                
                re_predict_map[(data_idx, len(input_ids) - 1)] = list(maped_entities[start_entity_idx]) + list(maped_entities[end_entity_idx])
                re_index += 1
        
        
    
    input_ids=np.array(input_ids, dtype=np.int32)
    input_type_ids = np.zeros((len(input_ids), max_len), dtype=np.int32)
    
    input_masks=np.array(input_masks, dtype=np.int32)
    input_entity_masks=np.array(input_entity_masks, dtype=np.int32)
    
    pos = generate_pos_ids(len(input_ids), max_len)
    
    x = [input_ids, input_type_ids, pos, input_masks, input_entity_masks]
    
    
    #预测关系
    y_predict = model.predict(x, batch_size=128, verbose=1)
    
    # re_predict构造
    re_predict = []
    
    for key, value in re_predict_map.items():
        data_idx = key[0]
        relation_idx = key[1]
        
        while data_idx >= len(re_predict):
            re_predict.append([])
        
        re = np.argmax(y_predict[0][relation_idx], axis=-1)
        
        if re >= 2: # 去除 null 和no relation
            re_predict[data_idx].append(tuple(value + [reid2type[re]]))
    
    re_f1 = eval_prf(re_predict, mapped_re_true)
    
    return ne_f1, re_f1
        