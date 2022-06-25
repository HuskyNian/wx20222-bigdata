import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoModel,AutoConfig
from transformers.models.bert.modeling_bert import  BertEmbeddings,BertPreTrainedModel
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
#from category_id_map import CATEGORY_ID_LIST
from collections import OrderedDict
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        #self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        #x = self.layer_norm(x)

        return x
class FinetuneModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.body = MultiModal(args)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.classifier =nn.Sequential(
          nn.BatchNorm1d(768*2),PositionwiseFeedForward(768*2,1024),nn.Linear(768*2,200))
        if args.pretrained_path is not None:
            print('load own pretrained model!')
            ck = torch.load(args.pretrained_path,map_location='cpu')['model']
            ck = OrderedDict({ k.replace('roberta.bert.',''):v  for k,v in ck.items() if 'roberta.bert.' in k})
            self.body.load_state_dict(ck)
        
    def forward(self,inputs,inference=False):
        bert_embedding,mask_copy = self.body(inputs,inference=inference)
        
        mean_last_hidden = bert_embedding['last_hidden_state']
        mean_last_hidden = (mean_last_hidden*mask_copy.unsqueeze(-1)).sum(1)/mask_copy.sum(1).unsqueeze(-1)
        
        #clf_token = bert_embedding['last_hidden_state'][:,0]

        mean_last4 = bert_embedding['hidden_states'][-4:]
        mean_last4 = sum( [(hidden*mask_copy.unsqueeze(-1)).sum(1)/mask_copy.sum(1).unsqueeze(-1) for hidden in mean_last4]   ) / 4

        final_embedding =torch.cat([mean_last_hidden,mean_last4],dim=1)
        
        prediction = self.classifier(final_embedding)
        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'],self.loss_fn)
    
    def load_my_state_dict(self,state_dict):
        own_state = self.state_dict()
        add_parts = []
        delete_parts = []
        loaded_names = []
        for name, param in state_dict.items():
            if name not in own_state:
                delete_parts.append(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            loaded_names.append(name)
        for i in own_state.keys():
            if i not in loaded_names:
                add_parts.append(i)
        print(f'delete parts: {delete_parts}\n new parts:{add_parts}')

    @staticmethod
    def cal_loss(prediction, label,loss_fn):#,level1_label,pred_level1,bce_fn):
        label = label.squeeze(dim=1)
        loss = loss_fn(prediction, label) 
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label,prediction
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)    

class MultiModal(BertPreTrainedModel):
    def __init__(self, args):
        
        # visual embeds
        self.config = config = AutoConfig.from_pretrained('hfl/chinese-roberta-wwm-ext')
        super().__init__(config)
        #config.vocab_size = 2
        self.visual_bert_embeddings = BertEmbeddings(config)
        self.visual_embeddings = nn.Sequential(nn.Linear(768,1024,bias=False),nn.ReLU(),
        nn.Linear(1024,1024,bias=False),nn.ReLU(),nn.Linear(1024,768,bias=False),nn.ReLU())
        self.init_weights()
        # fuse bert
        self.seq_len = args.bert_seq_length
        self.embed_fuse = PositionwiseFeedForward(768,1024)

        self.bert = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.bert.pooler = nn.Identity()
        self.encoder = self.bert.encoder

        # word embeds
        self.word_embeddings = self.bert.embeddings

        
        #self.word_embeddings = self.bert.embeddings.word_embeddings
        #self.token_ids = torch.tensor([0 if i<args.bert_seq_length else 1 for i in range(args.bert_seq_length+32)],device='cuda:0')

    def forward(self, inputs, inference=False):
        
        #token_type_ids = torch.tile(self.token_ids,(len(inputs['title_input']),1))
        masks = torch.cat([inputs['title_mask'],inputs['frame_mask']],dim=1)
        mask_copy = masks.clone()
        masks = masks[:,None,None,:]
        masks = (1.0-masks) * -10000.0

        word_embeds = self.word_embeddings(inputs['title_input'])
        
        visual_embeds = self.visual_embeddings(inputs['frame_input'])
        visual_embeds = self.visual_bert_embeddings(inputs_embeds=visual_embeds)

        all_feats = torch.cat([word_embeds,visual_embeds],dim=1)
        all_feats = self.embed_fuse(all_feats)
        assert all_feats.shape[1] ==self.seq_len+32 and all_feats.shape[2] ==768
        
        bert_embedding = self.encoder(all_feats,
         attention_mask=masks,head_mask=self.bert.get_head_mask(None,self.bert.config.num_hidden_layers),
         return_dict=self.bert.config.use_return_dict,use_cache=False,output_hidden_states=True)
        return bert_embedding,mask_copy

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding
