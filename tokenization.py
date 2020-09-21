from google_bert.tokenization import *
from google_bert.tokenization import _is_control, _is_whitespace
import numpy as np
import os

def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def _word_piece_tokenize(token, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    chars = list(token)
    if len(chars) > max_input_chars_per_word:
        return [unk_token], [0 for i in range(len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_tokens_offset = [0 for i in range(len(chars))]
    
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        for i in range(start,end):
            sub_tokens_offset[i] = len(sub_tokens) - 1
        start = end

    if is_bad:
        return [unk_token], [0 for i in range(len(chars))]
    else:
        return sub_tokens, sub_tokens_offset

class BERTTextEncoder(object):
    def __init__(self, vocab_file: str, do_lower_case: bool = True) -> None:
        self.do_lower_case = do_lower_case
        
        # vocab词表信息
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # 加载对应
        self.bert_pad_id = self.vocab['[PAD]']
        # 99 used
        self.bert_unk_id = self.vocab['[UNK]']
        self.bert_cls_id = self.vocab['[CLS]']
        self.bert_sep_id = self.vocab['[SEP]']
        self.bert_msk_id = self.vocab['[MASK]']
        self.vocab_size = len(self.vocab) - 99 - 5
        
        
    def standardize_ids(self, ids):
        for i in range(len(ids)):
            if ids[i] == self.bert_pad_id:  # PAD
                ids[i] = 1 + self.vocab_size
            elif ids[i] == self.bert_unk_id:  # UNK
                ids[i] = 0
            elif ids[i] == self.bert_cls_id:  # CLS
                ids[i] = 3 + self.vocab_size
            elif ids[i] == self.bert_sep_id:  # SEP
                ids[i] = 5 + self.vocab_size
            elif ids[i] == self.bert_msk_id:  # MASK
                ids[i] = 2 + self.vocab_size
            elif ids[i] > self.bert_msk_id:  # VOCAB
                ids[i] -= self.bert_msk_id
        return ids

    # 返回 tokens, real_token_offset
    def tokenize(self, strA):
        orig_tokens = []
        char_orig_token_offset = []
        
        is_token_start = True
        is_token_end = False
        # 初步分词
        for char in strA:
            cp = ord(char)
            
            if cp == 0 or cp == 0xfffd or _is_control(char):
                char = ''
            elif _is_whitespace(char):
                is_token_end = True
                char = ''
            elif _is_chinese_char(char):
                is_token_start = True
                is_token_end = True
                
                
            if is_token_start:
                if len(orig_tokens) == 0 or len(orig_tokens[-1]) != 0: 
                    orig_tokens.append('')
                is_token_start = False
            
            if char == '':
                char_orig_token_offset.append(None)
            else:
                char_orig_token_offset.append((len(orig_tokens) - 1, len(orig_tokens[-1])))
                orig_tokens[-1] += char
        
            if is_token_end:
                is_token_start = True
                is_token_end = False
        
        if orig_tokens[-1] == '':
            orig_tokens.pop()
        
        
        # 详细分词
        
        split_tokens = []
        split_tokens_map = {}
        
        for idx, token in enumerate(orig_tokens):
            if self.do_lower_case:
                token = token.lower()
                token = _run_strip_accents(token)
            
            sub_tokens, sub_tokens_offset = _word_piece_tokenize(token, self.vocab)
            
            for o_idx, offset in enumerate(sub_tokens_offset):
                split_tokens_map[(idx, o_idx)] =  len(split_tokens) + offset
            
            split_tokens.extend(sub_tokens)
            
        
        real_token_offset = []
        
        for offset in char_orig_token_offset:
            if offset is None:
                real_token_offset.append(None)
            elif offset in split_tokens_map:
                real_token_offset.append(split_tokens_map[offset])
            else:
                print("error in" + strA)
                
        
        return split_tokens, real_token_offset
    
    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)