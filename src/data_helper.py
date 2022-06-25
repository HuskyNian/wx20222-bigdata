from email import generator
import json
import random
import zipfile
from io import BytesIO
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer,AutoTokenizer
import jieba
import gc
import pickle
from category_id_map import category_id_to_lv2id
from sklearn.model_selection import StratifiedKFold

title_max_len = 50
asr_max_len = 100
ocr_max_len = 102
min_title_len = 80
#min_asr_len = 106
#min_ocr_len = 106
min_asr_len = 86
min_ocr_len = 86

def make_y(anns,visual_feats,test=False):
    y = []
    for i in range(len(anns)):
        anns[i]['visual_feats'] = visual_feats[i]
        all_ocr = ''
        ocrs = anns[i]['ocr']
        for ocr in ocrs:
            all_ocr += ocr['text']
        anns[i]['ocr'] = all_ocr
        if not test:
            label = anns[i].pop('category_id')
            y.append( category_id_to_lv2id(label) )
    return anns ,y
# cls + s1 + sep + cls + s2 + sep + cls + s3 + sep
def cut_min(s,this_len,need_cut,min_len):
    if need_cut < (this_len - min_len):  # e.g.  100 < 500 - 150
        mid = (this_len - need_cut)//2 # e.g.   200 = (500-100)/2
        s['input_ids'] = s['input_ids'][:mid] + s['input_ids'][-mid:]
        s['attention_mask'] = s['attention_mask'][:mid] + s['attention_mask'][-mid:]
        return s,-1
    else:
        mid = min_len // 2
        s['input_ids'] = s['input_ids'][:mid] + s['input_ids'][-mid:]
        s['attention_mask'] = s['attention_mask'][:mid] + s['attention_mask'][-mid:]
        return s, need_cut - (this_len - min_len)

def combine_text(anns,tokenizer):
    for an in tqdm(anns): 
        s1 = tokenizer(an['title'])
        s2 = tokenizer(an['asr'])
        s3 = tokenizer(an['ocr'])

        l1 = len(s1['input_ids'])
        l2 = len(s2['input_ids'])
        l3 = len(s3['input_ids'])
        #print(f'origin len:{l1} {l2} {l3}')
        need_cut = l1 +l2 +l3 - 258 # 186
        
        if need_cut >0 and l2>min_asr_len:
            s2,need_cut = cut_min(s2,l2,need_cut,min_asr_len)
        if need_cut >0 and l3>min_ocr_len:
            s3,need_cut = cut_min(s3,l3,need_cut,min_ocr_len)
        if need_cut >0 :
            s1,need_cut = cut_min(s1,l1,need_cut,min_title_len)

        input_ids = s1['input_ids'] + s2['input_ids'][1:] + s3['input_ids'][1:]
        pad = 256 - len(input_ids)
        input_ids = input_ids + [0 for i in range(pad)]
        attention_mask = s1['attention_mask'] + s2['attention_mask'][1:] + s3['attention_mask'][1:] + [0 for i in range(pad)]
        if len(input_ids) ==530:
            print(f'error {l1,l2,l3}')
            print('wo shi zhu',len(s1['input_ids']),len(s2['input_ids']),len(s3['input_ids']))
        '''try:
            assert len(input_ids) ==256 and len(attention_mask)==256
        except:
            print(f'{len(input_ids)},{len(attention_mask)}')
            assert False'''
        an['input_ids'] = input_ids
        an['attention_mask'] = attention_mask
    return anns

def create_dataloaders(args,g,fold = 0):
    with open(args.train_annotation, 'r',) as f:
        anns = json.load(f)
    visual_feats = np.load(args.train_zip_feats,allow_pickle=True)
    print('data loaded!')
    anns,y= make_y(anns,visual_feats)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
    anns = combine_text(anns,tokenizer)
    del visual_feats
    gc.collect()
    X_train, X_test, y_train, y_test = train_test_split(anns, y, test_size=args.val_ratio,
     random_state=args.seed,stratify=y)

    '''
    # sk fold
    print('using k fold to split data')
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    this_fold = 0
    for train_index,test_index in skf.split(anns,y):
        if this_fold == fold:
            break
        this_fold +=1
    anns = np.array(anns,dtype=object)
    y = np.array(y)
    X_train, X_test, y_train, y_test = anns[train_index],anns[test_index],y[train_index],y[test_index]
    '''
    print('data splited!')
    #train_dataset = MultiModalDataset(args,X_train,y_train)
    train_dataset = MultiModalDataset(args,anns,y)
    val_dataset = MultiModalDataset(args,X_test,y_test) 
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  pin_memory=True,generator=g,
                                  num_workers=args.num_workers,)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                batch_size=args.batch_size,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,)
    

    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    def __init__(self,
                 args,
                 anns,
                 y=None,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        self.anns = anns
        self.y = y
        if y is None and test_mode==False:
            print('must have y in train')
            assert False
        #self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
  

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        raw_feats = self.anns[idx]['visual_feats']
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    '''def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask'''

    def __getitem__(self, idx: int) -> dict:
        frame_input, frame_mask = self.get_visual_feats(0, idx)
        # Step 2, load title tokens
        '''title = self.anns[idx]['title'].replace(' ','')
        asr =  self.anns[idx]['asr'].replace(' ','')
        ocr = self.anns[idx]['ocr'].replace(' ','')
        l1 = len(title)
        l2 = len(asr)
        l3 = len(ocr)
        
        if l1 + l2 + l3>self.bert_seq_length-4:
            if l1>50:
                title =title[:title_max_len]
            if len(title) + l2 >self.bert_seq_length-104:
                title = title[:title_max_len]
                asr = asr[:asr_max_len]
        text = title+'[SEP]'+asr + '[SEP]' + ocr      
        text = text[:self.bert_seq_length-2]
        title_input, title_mask = self.tokenize_text(text)'''
        title_input,title_mask = torch.LongTensor(self.anns[idx]['input_ids']),torch.LongTensor(self.anns[idx]['attention_mask'])
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = self.y[idx]
            data['label'] = torch.LongTensor([label])

        return data

