import torch
from torch.utils.data import SequentialSampler, DataLoader
from transformers import AutoTokenizer
from config import parse_args
from data_helper import MultiModalDataset,combine_text,make_y
from category_id_map import lv2id_to_category_id
from model import FinetuneModel
from tqdm import tqdm
import json
import torch.nn as nn
import numpy as np
import os
def inference():
    args = parse_args()
    print(f'testing {args.test_zip_feats} {args.test_annotation}')
    # 1. load data
    with open(args.test_annotation, 'r',) as f:
        anns = json.load(f)
    visual_feats = np.load(args.test_zip_feats,allow_pickle=True)
    print('data loaded!')
    anns,y= make_y(anns,visual_feats,test=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
    anns = combine_text(anns,tokenizer)
    dataset = MultiModalDataset(args, anns, test_mode=True)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,)
                            #prefetch_factor=args.prefetch)

    # 2. load model
    root = '/content/drive/MyDrive/weixin_bigdata/data/saved_models/'
    '''model_paths = ['model_epoch_3_mean_f1_0.6905newfold0.bin',
                  'model_epoch_3_mean_f1_0.6779newfold1.bin',
                  #'model_epoch_4_mean_f1_0.6783newfold2.bin'
                  'model_epoch_3_mean_f1_0.6853newfold3.bin',
                  'model_epoch_4_mean_f1_0.6787newfold4.bin',
                  #'model_epoch_2_mean_f1_0.6755newfold5.bin',
                  'model_epoch_3_mean_f1_0.6843newfold6.bin',
                  ]'''
    model_paths =[
      'model_epoch_3_mean_f1_0.6898newfold0.bin',
      'model_epoch_3_mean_f1_0.6782newfold1.bin',
      'model_epoch_3_mean_f1_0.6803newfold2.bin',
      'model_epoch_3_mean_f1_0.6848newfold3.bin',
      'model_epoch_3_mean_f1_0.6757newfold4.bin',
      'model_epoch_3_mean_f1_0.6753newfold5.bin',
      'model_epoch_3_mean_f1_0.6833newfold6.bin',
      'model_epoch_3_mean_f1_0.6843newfold7.bin',
      'model_epoch_3_mean_f1_0.6905newfold0.bin',
      'model_epoch_3_mean_f1_0.6779newfold1.bin',
      'model_epoch_4_mean_f1_0.6783newfold2.bin',
      'model_epoch_3_mean_f1_0.6853newfold3.bin',
      'model_epoch_4_mean_f1_0.6787newfold4.bin',
      'model_epoch_2_mean_f1_0.6755newfold5.bin',
      'model_epoch_3_mean_f1_0.6843newfold6.bin',
      'model_epoch_3_mean_f1_0.9354newfold42.bin',
      'model_epoch_3_mean_f1_0.9496newfold111.bin',
      'model_epoch_3_mean_f1_0.6805newfold999.bin',
      'model_epoch_3_mean_f1_0.8744newfold444.bin',
      'model_epoch_3_mean_f1_0.8724newfold333.bin',
      'model_epoch_3_mean_f1_0.8745newfold222.bin',
    ]
    model_paths = os.listdir('../finetune_models')
    print(f'loading models{model_paths}')
    models = []
    for path in model_paths:
        model =FinetuneModel(args)
        model.body.visual_bert_embeddings.word_embeddings = nn.Identity()
        checkpoint = torch.load(root+path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        models.append(model)

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred_all = None
            for model in models:
                pred_label_id = model(batch, inference=True)
                if pred_all is None:
                    pred_all = pred_label_id
                else:
                    pred_all += pred_label_id
            pred_all = torch.argmax(pred_all,dim=1)
            predictions.extend(pred_all.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    print('inference done!')


if __name__ == '__main__':
    inference()
