import torch
from torch.utils.data import SequentialSampler, DataLoader
from transformers import AutoTokenizer
from config import parse_args
from data_helper import MultiModalDataset,combine_text,make_y
from category_id_map import lv2id_to_category_id
from model import FinetuneModel
from tqdm import tqdm
import torch.nn as nn
import json
import numpy as np
def inference():
    args = parse_args()
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
    model =FinetuneModel(args)
    model.body.visual_bert_embeddings.word_embeddings = nn.Identity()
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred_label_id = model(batch, inference=True)
            pred_label_id = torch.argmax(pred_label_id,dim=1)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    print('inference done!')


if __name__ == '__main__':
    inference()
