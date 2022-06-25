import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup

from category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def build_new_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    print('building new optimizer')
    no_decay = ['bias', 'LayerNorm.weight']
    if not args.albef:
        model_lr = {'body':5e-5,'classifier':5e-4}
        optimizer_grouped_parameters = []
        for layer_name in model_lr:
            lr = model_lr[layer_name]
            optimizer_grouped_parameters += [
                {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                      and layer_name in n)],
                'weight_decay': args.weight_decay,
                'lr':lr},
                {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                      and layer_name in n)], 'weight_decay': 0.0,
                'lr':lr}
            ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=model_lr['body'], weight_decay = args.weight_decay)
        print(f'setting lr:{model_lr},')
    else:
        lr = args.learning_rate 
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += [
                {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay))],
                'weight_decay': args.weight_decay,
                },
                {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay))], 'weight_decay': 0.0,
                }
            ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay = args.weight_decay)
        print(f'setting lr:{lr},')
    
    warmup_steps = 1000 #int(args.max_epochs * (100000/args.batch_size)*0.1)
    print(f'warmup steps{warmup_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=args.max_steps)

    return optimizer, scheduler

def build_pretrain_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    print('building new optimizer')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    
    layer_name = 'roberta.bert'
    lr = args.pretrain_learning_rate
    optimizer_grouped_parameters += [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                and layer_name in n)],
          'weight_decay': args.weight_decay,
        'lr':lr},
        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                              and layer_name in n)], 'weight_decay': 0.0,
        'lr':lr}
    ]
    
    optimizer_grouped_parameters += [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                and layer_name not in n)],
          'weight_decay': args.weight_decay,
        'lr':lr},
        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                              and layer_name not in n)], 'weight_decay': 0.0,
        'lr':lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.pretrain_learning_rate, weight_decay = args.weight_decay)
    
    
    total_steps = int(args.pretrain_epochs * (1125000/args.batch_size))
    warmup_steps = int(total_steps*0.06)
    print(f'setting lr:{args.pretrain_learning_rate} ,total_steps{total_steps},warm up{warmup_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    return optimizer, scheduler

def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.pretrain:
        lr = args.pretrain_learning_rate
        warmup_steps = 1000
        max_steps = args.max_steps*7
    else:
        lr = args.learning_rate
        warmup_steps = args.warmup_steps
        max_steps = args.max_steps
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay = args.weight_decay)
    if args.weight is not None:
        args.warmup_steps = 0
        assert args.warmup_steps == 0
    print(f'setting lr:{lr},max_steps{max_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=max_steps)

    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results
