import json
from data_helper import combine_text
min_title_len = 80
min_asr_len = 86
min_ocr_len = 86
def combine_ocr(anns,):
    for i in range(len(anns)):
        all_ocr = ''
        ocrs = anns[i]['ocr']
        for ocr in ocrs:
            all_ocr += ocr['text']
        anns[i]['ocr'] = all_ocr
    return anns 

with open('../data/annotations/unlabeled.json', 'r',) as f:
    anns = json.load(f)
len(anns)
from transformers import AutoTokenizer
import gc
import random
import torch
from tqdm import tqdm
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

anns = combine_ocr(anns)
anns = combine_text(anns,tokenizer)
for i in range(20):
    this = anns[i*50000:(i+1)*50000]
    for item in tqdm(this):
        this_id = item['id']
        visual_feats = np.load(f'../data/zip_feats/unlabeled/{this_id}.npy')
        item['visual_feats'] = visual_feats
    this = np.array(this,dtype=object)
    np.save(f'../data/zip_feats/clustered_feats/{i}.npy',this)  

## adding training and test text
import json
with open('../data/annotations/labeled.json', 'r',) as f:
    anns_train = json.load(f)
with open('../data/annotations/test_b.json', 'r',) as f:
    anns_test = json.load(f)   
anns_train = combine_ocr(anns_train)
anns_train = combine_text(anns_train,tokenizer)
anns_test = combine_ocr(anns_test)
anns_test = combine_text(anns_test,tokenizer)
visual_feats = np.load('../data/labeled.npy',allow_pickle=True)
for i in tqdm(range(len(anns_train))):
   
    anns_train[i]['visual_feats'] = visual_feats[i]
visual_feats = np.load('../data/testb.npy',allow_pickle=True)
for i in tqdm(range(len(anns_test))):
   
    anns_test[i]['visual_feats'] = visual_feats[i]
anns = anns_train + anns_test
this = anns[:60000]
this = np.array(this,dtype=object)
np.save(f'../data/zip_feats/clustered_feats/20.npy',this)     
this = anns[60000:]
this = np.array(this,dtype=object)
np.save(f'../data/zip_feats/clustered_feats/21.npy',this)     