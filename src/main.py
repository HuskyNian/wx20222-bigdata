import logging
import os
import time
import torch
import gc
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal,FinetuneModel
from util import setup_device, setup_seed, setup_logging, \
      build_optimizer, evaluate,build_new_optimizer
import random
import numpy as np
import torch.nn as nn
import json
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel, SWALR
from albef import ALBEF
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.5, alpha=0.3, emb_name='bert.embeddings.word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='bert.embeddings.word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def seed_everything(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    def seed_worker(worker_id):
        global seed
        np.random.seed(seed)
        random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(1)
    return g

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label,prediction = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args,fold=0,seed=2022,name = 'common91'):
    # 1. load data
    print(f'train')
    start_time = time.time()
    g= seed_everything(2022)
    #g=seed_everything(args.seed)
    train_dataloader, val_dataloader = create_dataloaders(args,g,fold=fold)
    print(f'dataloader created! after{time.time()-start_time}')
    # 2. build model and optimizers
    model = FinetuneModel(args)
    
    if args.weight is not None:
        print(f'load model from:{args.weight}')
        checkpoint = torch.load(args.weight, map_location='cpu')
        model.load_my_state_dict(checkpoint['model_state_dict'])
    model.body.visual_bert_embeddings.word_embeddings = nn.Identity()

    #model = ALBEF(args)
    optimizer, scheduler = build_new_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    ema = ModelEma(model,args.ema_decay,'cuda')
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer,anneal_strategy='linear',
                          anneal_epochs=9,swa_lr=0.05)#here epochs is step
    print('start training')
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    pgd = PGD(model)
    K=2
    swa_step = 0
    swa_start = int(args.max_epochs * (90000/args.batch_size)*0.75)
    swa_gap = int((args.max_epochs * (90000/args.batch_size) - swa_start)/9)
    print(f'swa will start from {swa_start} with gap{swa_gap}')
    for epoch in range(args.max_epochs):
        for batch_idx,batch in enumerate(train_dataloader):
            model.train()
            loss, accuracy, _, _,prediction = model(batch)
            loss = loss.mean() /args.gradient_acc_step
            accuracy = accuracy.mean()
            loss.backward()
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv, accuracy, _, _,_ = model(batch)
                loss_adv = loss_adv.mean()/args.gradient_acc_step
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()
            if ((batch_idx + 1) % args.gradient_acc_step == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                ema.update(model)
            if step+1> swa_start and (step+1-swa_start)%swa_gap==0:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                swa_step += 1
                print(f'swa is {swa_step} time')
            scheduler.step()
            #if step+1==8000 or step+1==9000+1125:
            #    validate_and_save(model,val_dataloader,step,epoch,ema,args,fold=fold)
            step += 1
            #if step == 9000+1125:
            #    break
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
        # 4. validation
    validate_and_save(model,val_dataloader,step,epoch,ema,args,fold=fold,name=name)

    '''torch.optim.swa_utils.update_bn(train_dataloader,swa_model,)
    loss, results = validate(swa_model, val_dataloader)
    results = {k: round(v, 4) for k, v in results.items()}
    print(f"Epoch {epoch} step {step}: swa loss {loss:.3f}, {results}")
    swa_f1 = results['mean_f1']
    if swa_f1>0.673:
        torch.save({'epoch': epoch, 'model_state_dict': swa_model.module.module.state_dict(), 'mean_f1': swa_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{swa_f1}newfold{fold}.bin')'''

def validate_and_save(model,val_dataloader,step,epoch,ema,args,fold=0,name='common91'):
    loss, results = validate(ema.module, val_dataloader)
    results = {k: round(v, 4) for k, v in results.items()}
    print(f"Epoch {epoch} step {step}: ema loss {loss:.3f}, {results}")
    ema_f1 = results['mean_f1']
    torch.cuda.empty_cache()
    gc.collect()
    loss, results = validate(model, val_dataloader)
    results = {k: round(v, 4) for k, v in results.items()}
    print(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

    torch.save({'epoch': epoch, 'model_state_dict': ema.module.module.state_dict(), 'mean_f1': ema_f1},
                    f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{ema_f1}newfold{fold}_{name}.bin')

    # 5. save checkpoint
    '''mean_f1 = results['mean_f1']
    if mean_f1 > 0.673 or ema_f1 >0.673:
        if ema_f1 >mean_f1:
            torch.save({'epoch': epoch, 'model_state_dict': ema.module.module.state_dict(), 'mean_f1': ema_f1},
                    f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{ema_f1}newfold{fold}.bin')
        else:
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                    f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}newfold{fold}.bin')'''

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    train_and_validate(args,fold=0,seed=2022,name = 'common91')
    train_and_validate(args,fold=1,seed=2022,name = 'common91')
    train_and_validate(args,fold=2,seed=2022,name = 'common91')
    train_and_validate(args,fold=3,seed=2022,name = 'common91')
    train_and_validate(args,fold=4,seed=2022,name = 'common91')
    train_and_validate(args,fold=5,seed=2022,name = 'common91')
    train_and_validate(args,fold=6,seed=2022,name = 'common91')
    train_and_validate(args,fold=7,seed=2022,name = 'common91')
    args.albef = False
    args.label_smoothing = 0.1
    train_and_validate(args,fold=0,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=1,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=2,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=3,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=4,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=5,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=6,seed=2022,name = 'common91_lr_sm')
    train_and_validate(args,fold=7,seed=2022,name = 'common91_lr_sm')

    train_and_validate(args,fold=0,seed=42,name = 'all')
    train_and_validate(args,fold=1,seed=111,name =  'all')
    train_and_validate(args,fold=2,seed=222,name =  'all')
    train_and_validate(args,fold=3,seed=333,name =  'all')
    train_and_validate(args,fold=4,seed=444,name =  'all')

    logging.info("Training/evaluation parameters: %s", args)
    with open(args.savedmodel_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    

if __name__ == '__main__':
    main()
