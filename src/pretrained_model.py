import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer
import numpy as np
import random 
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os 
import json
from model import MultiModal
from config import parse_args
from util import setup_device, setup_seed, build_pretrain_optimizer, evaluate
from main import ModelEma
from data_helper import MultiModalDataset
import gc
import time
from copy import deepcopy

class MaskLM(object):
    def __init__(self, tokenizer_path='hfl/chinese-roberta-wwm-ext', mlm_probability=0.25):
        self.mlm_probability = 0.25
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ] #  cls 我是谁 sep => 1 0 0 0 1
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
class MaskVideo(object):
    def __init__(self, mlm_probability=0.25):
        self.mlm_probability = 0.25
        
    def torch_mask_frames(self, video_feature, video_mask):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mlm_probability)
        probability_matrix = probability_matrix * video_mask
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2)) 

        return inputs, video_labels_index
class ShuffleVideo(object):
    def __init__(self):
        pass
    
    def torch_shuf_video(self, video_feature,mfm_label):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        mfm_label = mfm_label[shuf_index]
        return video_feature, label,mfm_label


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
class UniBertForMaskedLM(nn.Module):
    def __init__(self, args,config):
        super().__init__()
        self.bert = MultiModal(args)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, return_mlm=False):
        inputs = {}
        inputs['title_input'] = text_input_ids
        inputs['title_mask'] = text_mask
        inputs['frame_input'] = video_feature
        inputs['frame_mask'] = video_mask
        encoder_outputs,_ = self.bert(inputs) # return embedding, mask_copy
        encoder_outputs = encoder_outputs['last_hidden_state']
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, :256 , :]
        else:
            return encoder_outputs, None        
class WXPretrainedModel(nn.Module):
    def __init__(self, args, task=['mlm', 'mfm','itm'], init_from_pretrain=True):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'{args.bert_dir}')
        self.task = set(task)
        self.celoss = nn.CrossEntropyLoss()
        self.bceloss = nn.BCEWithLogitsLoss()
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            self.vocab_size = uni_bert_cfg.vocab_size
            print(f'now using mlm rate{self.lm.mlm_probability}')
        
        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg) 
            print(f'now using mfm rate{self.vm.mlm_probability}')
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 
            print('training itm')

        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM(args, config=uni_bert_cfg)
        else:
            print('model not initiated from pretrained')
            assert False
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, target=None, task=None):
        loss, pred = 0, None
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            #lm_label = lm_label[:, 1:].to(text_input_ids.device) # [SEP] 卡 MASK 大师 [SEP]
            lm_label = lm_label.to(text_input_ids.device)
            return_mlm = True
            
        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            # inputs masked, label: bs * max_frame
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label,video_label = self.sv.torch_shuf_video(video_feature.cpu(),video_label)
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)
            
        # concat features
        features, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)
        
        # compute pretrain task loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = self.celoss(pred, lm_label.contiguous().view(-1))
            masked_lm_loss = masked_lm_loss / 1.25 / len(sample_task)
            loss += masked_lm_loss
        masked_vm_loss = 0 
        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, 256:, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     video_mask, video_label, normalize=False)
            masked_vm_loss = masked_vm_loss / 3 / len(sample_task)
            loss += masked_vm_loss
        itm_loss = 0 
        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = self.bceloss(pred.view(-1), video_text_match_label.view(-1))
            itm_loss= itm_loss / len(sample_task)
            #itm_loss= itm_loss / 100/ len(sample_task)
            loss += itm_loss
            
        return (pred, loss, masked_lm_loss.item(),masked_vm_loss,itm_loss)

    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
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
def create_pretrained_dataloaders(args,g,f,f1):
    # todo : choose one anns to load
    
    anns = np.load(f,allow_pickle=True)
    anns1 = np.load(f1,allow_pickle=True)
    anns = np.concatenate([anns, anns1],axis=0 )
    del anns1
    gc.collect()
    
    print('data loaded!')
    X_test =anns[:1000]
    print('data splited!')
    train_dataset = MultiModalDataset(args,anns,test_mode=True)
    val_dataset = MultiModalDataset(args,X_test,test_mode=True) 
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  pin_memory=True,generator=g,
                                  num_workers=args.num_workers,)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                batch_size=args.val_batch_size,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,)
    return train_dataloader, val_dataloader 
DEVICE = 'cuda'
def get_pred_and_loss(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_input'].to(DEVICE)
    input_ids = item['title_input'].to(DEVICE)
    attention_mask = item['title_mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)
    
    pred, loss,lm_loss,vm_loss,itm_loss = model(video_feature, video_mask, input_ids, attention_mask)
    return pred, loss,lm_loss,vm_loss,itm_loss
def start_pretrain():
    
    g = seed_everything(2022)

    args = parse_args()
    setup_device(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    
    model = WXPretrainedModel(args)
    if args.pretrained_path is not None:
        print('pretrain load pretrained model')
        ck = torch.load(args.pretrained_path,map_location='cpu')
        model.load_state_dict(ck['model'],strict=False)
    model = model.cuda()

    #ema = ModelEma(model,args.ema_decay,device='cuda')
    optimizer, scheduler =  build_pretrain_optimizer(args, model)
    #scheduler.load_state_dict(ck['scheduler'])
    start_time = time.time()
    
    step = 0
    
    
    for epoch in range(args.pretrain_epochs):
        train_files = os.listdir(args.pretrain_datapath)
        train_files = [args.pretrain_datapath +i for i in train_files]
        half_num = len(train_files) //2
        random.shuffle(train_files)
        print(f'now have {half_num*2} training files')
        print(f'scheduler {scheduler.state_dict()}')
        for file_num in range(half_num): # 22 npy files in total ############ 改成11
            # build loaders
            try:
                del train_dataloader,val_dataloader
                gc.collect()
            except:
                print('no loader to delete')
            f1,f2 = train_files[file_num],train_files[file_num+half_num]
            train_dataloader,val_dataloader = create_pretrained_dataloaders(args,g,f1,f2)
            num_total_steps = len(train_dataloader) *half_num* args.pretrain_epochs
            for batch_idx,batch in enumerate(train_dataloader):
                model.train()
                pred, loss,lm_loss,vm_loss,itm_loss = get_pred_and_loss(model,batch)
                loss = loss /args.gradient_acc_step
                loss.backward()
                if ((batch_idx + 1) % args.gradient_acc_step == 0) or (batch_idx + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    #ema.update(model)
                
                scheduler.step()
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    print(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, lm_loss {lm_loss:.3f},vm_loss {vm_loss:.3f}itm_loss {itm_loss:.3f}")
                step+=1
            # 4. validation
            optimizer.step()
            optimizer.zero_grad()
            model.eval()
            losses = []
            lm_losses = []
            vm_losses  = []
            itm_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    pred, loss,lm_loss,vm_loss,itm_loss = get_pred_and_loss(model,batch)
                    losses.append(loss.cpu().numpy())
                    lm_losses.append(lm_loss)
                    vm_losses.append(vm_loss)
                    itm_losses.append(itm_loss)
            loss = sum(losses) / len(losses)
            lm_loss = sum(lm_losses) / len(losses)
            vm_loss = sum(vm_losses) / len(losses)
            itm_loss = sum(itm_losses) / len(losses)
            print(f'total loss: {loss:.3f},lm_loss:{lm_loss:.3f},vm_loss:{vm_loss:.3f},itm_loss:{itm_loss:.3f}')
            if file_num == 3 or file_num == 6:
                ckpt = {'model':model.state_dict(),
                   'epoch':epoch,
                   'scheduler':scheduler.state_dict(),
                   }
                torch.save(ckpt,args.savedmodel_path + '/last.bin')    
                print('model save !')
        ckpt = {'model':model.state_dict(),
                   'epoch':epoch,
                   'scheduler':scheduler.state_dict(),
                   }
        torch.save(ckpt,args.savedmodel_path + '/last.bin')    
        print('model save !')
if __name__ == "__main__":
    start_pretrain()