# GPU指定
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


import tensorflow as tf
from tqdm import tqdm
import json
import datetime,time
import os
import shutil

from keras.utils import to_categorical
from keras.models import *
from keras import backend as K

from keras.layers import *
from keras.layers import Layer

from keras.optimizers import *
from keras.callbacks import Callback, CSVLogger

from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.layers import CRF

from transformer.load import load_google_bert
from data.dataset import generate_pos_ids
from evaluation import eval_model

# 导入常量
from config import *

# 数据路径
data_path = './data_set/'


# 文本最大长度
max_len = 64
# 集中注意力层数量
focused_attention_mask_num = 4

# 训练参数
epochs = 80
batch_size = 48

## 联合学习分数比例
joint_model_loss_weights={
    'score_dense': 1.,
    'crf': 1.,
}

## 优化器
optimizer_class = Adam
optimizer_config = {
    "lr": 5e-5, 
    "beta_1": 0.9, 
    "beta_2": 0.999, 
    "epsilon": 1e-6, 
    "decay": 0.01
}


# 载入原始数据
train_data = json.load(open(data_path+"train.json"))
dev_data = json.load(open(data_path+"dev.json"))
test_data = json.load(open(data_path+"test.json"))


print(len(train_data))
print(len(dev_data))
print(len(test_data))


def re_data_to_input_output(model_data, bert_encoder, max_len):
    
    # 输入构造
    input_ids = []
    input_masks = []
    input_entity_masks = []
    
    output_re_ids = []
    output_ne_ids = []
    
    for item in tqdm(model_data):
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
        
        relation_map = {(start_entity_idx, end_entity_idx):relation_type for start_entity_idx, end_entity_idx, relation_type in relations}
    
        netype_list = ['O' for _ in range(len(tokens))]
        
        for start_pos, end_pos, entity_type in maped_entities:
            if end_pos - start_pos == 1:
                netype_list[start_pos] = "S-" + entity_type
            else:
                netype_list[start_pos] = "B-" + entity_type
                netype_list[end_pos - 1] = "E-" + entity_type
                for tmp_pos in range(start_pos + 1, end_pos - 1):
                    netype_list[tmp_pos] = "I-" + entity_type
            

        
        for start_entity_idx in range(len(entities)):
            for end_entity_idx in range(start_entity_idx + 1, len(entities)):
                remask = [0 for _ in range(len(tokens))]
                
                for tmp_pos in range(maped_entities[start_entity_idx][0], maped_entities[start_entity_idx][1]):
                    remask[tmp_pos] = 1
                
                for tmp_pos in range(maped_entities[end_entity_idx][0], maped_entities[end_entity_idx][1]):
                    remask[tmp_pos] = 1
                
                retype = ""
                if (start_entity_idx, end_entity_idx) in relation_map:
                    retype = relation_map[(start_entity_idx, end_entity_idx)]
                else:
                    retype = 'no_relation'
                
                
                input_tokens = ['[CLS]'] + tokens + ['[SEP]'] 
                
                input_id = bert_encoder.standardize_ids(bert_encoder.convert_tokens_to_ids(input_tokens))
                input_mask = [1] * len(input_tokens)
                input_entity_mask = [1] + remask + [0]
                
                output_re_id = retype2id[retype]
                
                output_netype_list = ['null'] + netype_list + ['null']
                output_ne_id = [netype2id[item] for item in output_netype_list]
                
                input_id += [0] * (max_len - len(input_id))
                input_mask += [0] * (max_len - len(input_mask))
                input_entity_mask += [0] * (max_len - len(input_entity_mask))
                output_ne_id += [0] * (max_len - len(output_ne_id))
                
                input_ids.append(input_id)
                input_masks.append(input_mask)
                input_entity_masks.append(input_entity_mask)
                output_re_ids.append(output_re_id)
                output_ne_ids.append(output_ne_id)
                
    
    input_ids=np.array(input_ids, dtype=np.int32)
    input_masks=np.array(input_masks, dtype=np.int32)
    input_entity_masks=np.array(input_entity_masks, dtype=np.int32)
    output_re_ids=np.array(output_re_ids, dtype=np.int32)
    output_ne_ids=np.array(output_ne_ids, dtype=np.int32)
        
    pos = generate_pos_ids(len(input_ids), max_len)
    input_type_ids = np.zeros((len(input_ids), max_len), dtype=np.int32)
    
    return [input_ids, input_type_ids, pos, input_masks, input_entity_masks], [output_re_ids, output_ne_ids]
    

x_train, y_train = re_data_to_input_output(train_data, bert_encoder, max_len)


y_train[0] = to_categorical(y_train[0], len(reid2type))
y_train[1] = y_train[1][:,:,np.newaxis]


print(x_train[0].shape)
print(x_train[1].shape)
print(x_train[2].shape)
print(x_train[3].shape)
print(x_train[4].shape)


print(y_train[0].shape)
print(y_train[1].shape)


# # 搭建模型


bert_model = load_google_bert(bert_parameter_location, max_len = max_len, use_diff_attention_mask=True)



# 模型封装

bert_dmask_token_input = Input(shape=(max_len, ), name='bert_dmask_token_input', dtype='int32')
bert_dmask_segment_input = Input(shape=(max_len, ), name='bert_dmask_segment_input', dtype='int32')
bert_dmask_position_input = Input(shape=(max_len, ), name='bert_dmask_position_input', dtype='int32')
bert_dmask_attention_mask_input = Input(shape=(max_len, ), name='bert_dmask_attention_mask_input', dtype='float32')
bert_dmask_focused_attention_mask_input = Input(shape=(max_len, ), name='bert_dmask_focused_attention_mask_input', dtype='float32')

bert_dmask_output = bert_model([bert_dmask_token_input, bert_dmask_segment_input, bert_dmask_position_input] + [bert_dmask_attention_mask_input] * (12 - focused_attention_mask_num) + [bert_dmask_focused_attention_mask_input] * focused_attention_mask_num)

bert_dmask_model = Model(inputs=[bert_dmask_token_input, bert_dmask_segment_input, bert_dmask_position_input, bert_dmask_attention_mask_input, bert_dmask_focused_attention_mask_input], outputs=bert_dmask_output, name="bert_dmask_model")
# bert_dmask_model.summary()


class BertPooler(Layer):
    def __init__(self, **kwargs):
        super(BertPooler, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return inputs[:,0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
class BertMask(Layer):
    def __init__(self, max_len, **kwargs):
        self.supports_masking = True
        # 移除cls
        self.remove_cls_mask = K.constant(np.diag([0] + [1] * (max_len - 1)))
        super(BertMask, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask=None):
        # print(inputs, input_mask)
        return K.equal(K.dot(inputs[1], self.remove_cls_mask), 1)

    def call(self, inputs, mask=None):
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]



token_input = Input(shape=(max_len, ), name='token_input', dtype='int32')
segment_input = Input(shape=(max_len, ), name='segment_input', dtype='int32')
position_input = Input(shape=(max_len, ), name='position_input', dtype='int32')
attention_mask_input = Input(shape=(max_len, ), name='attention_mask_input', dtype='float32')
focused_attention_mask_input = Input(shape=(max_len, ), name='focused_attention_mask_input', dtype='float32')

bert_f_mask = bert_dmask_model([token_input, segment_input, position_input, attention_mask_input, focused_attention_mask_input])
bert_o_mask = bert_dmask_model([token_input, segment_input, position_input, attention_mask_input, attention_mask_input])

bert_pooler =  BertPooler(name='bert_pooler')(bert_f_mask)

pooler_dense = Dense(768, activation='tanh', name="pooler_dense")(bert_pooler)

score_dense = Dense(len(reid2type), activation='softmax', name="score_dense")(pooler_dense)

bert_mask = BertMask(max_len=max_len, name='bert_mask')([bert_o_mask, attention_mask_input])

crf =  CRF(len(neid2type), sparse_target=True, name='crf')(bert_mask)

model = Model(inputs=[token_input, segment_input, position_input, attention_mask_input, focused_attention_mask_input], outputs=[score_dense, crf], name="joint_model")
model.summary()



optimizer = optimizer_class(**optimizer_config)

joint_model_loss = {
    'score_dense': 'categorical_crossentropy',
    'crf': crf_loss,
}


model.compile(loss=joint_model_loss, loss_weights=joint_model_loss_weights, optimizer=optimizer)



class SaveModelBestCheckpoint(Callback):
    """自动保存最佳模型
    """
    def __init__(self, model_saved_path):
        self.model_saved_path = model_saved_path
        
        if not os.path.isdir(self.model_saved_path):
            os.makedirs(self.model_saved_path)
        
        self.best_score = None
    def on_epoch_end(self, epoch, logs=None):
        tmp_score = eval_model(dev_data, self.model, max_len)[1]
        
        if self.best_score is None or tmp_score > self.best_score:
            self.best_score = tmp_score
            self.model.save(os.path.join(self.model_saved_path, "best_weights"), overwrite=True)
    
    


class SaveModelLastCheckpoint(Callback):
    """自动保存最新模型
    """
    def __init__(self, model_saved_path):
        self.model_saved_path = model_saved_path
        
        if not os.path.isdir(self.model_saved_path):
            os.makedirs(self.model_saved_path)
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(os.path.join(self.model_saved_path, "last_weights"), overwrite=True)




print('relation extraction bert Train...')
now = time.strftime("%Y-%m-%d_%H-%M-%S")
projectPath = './param/outputModelWeights/{}'.format(projectName)
if not os.path.isdir(projectPath): os.makedirs(projectPath)
resultPath = projectPath + '/{}/'.format(now)
os.makedirs(resultPath)

callbacks = [
    SaveModelBestCheckpoint(resultPath),
    SaveModelLastCheckpoint(resultPath),
    CSVLogger(resultPath + 'training.log'),
]


model.fit(
    x_train, y_train, 
    epochs=epochs, batch_size=batch_size, callbacks=callbacks
)



print(resultPath)


# best_model = pathOutputModelWeights
model.load_weights(resultPath + "best_weights")


eval_model(test_data, model, max_len)



