from functools import partial
from xbert import BertConfig, BertModel
from transformers import AutoModel
from transformers.models.bert.modeling_bert import  BertEmbeddings,BertPreTrainedModel
import torch
import json
from torch import nn
from model import PositionwiseFeedForward
import torch.nn.functional as F
config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 21128,
  "fusion_layer": 6,
  "encoder_width": 768
}
with open('bert_config.json', 'w') as f:
    json.dump(config, f)

class ALBEF(BertPreTrainedModel):
    def __init__(self,                 
                 config = None,     
                 ):
        
         
        self.distill = config.distill
        
        ## video encoder
        visual_config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext')    # Download configuration from S3 and cache.
        super().__init__(visual_config)
        visual_config.num_hidden_layers = 1
        self.visual_encoder = AutoModel.from_config(visual_config)
        self.visual_encoder.embeddings.word_embeddings = nn.Identity()
        self.visual_encoder.pooler = nn.Identity()
        #self.visual_encoder._init_weights(self.visual_encoder)
        #self.visual_encoder = nn.Sequential(nn.Linear(768,1024,bias=False),nn.ReLU(),
        #nn.Linear(1024,1024,bias=False),nn.ReLU(),nn.Linear(1024,768,bias=False),nn.ReLU())
        self.visual_cls = nn.Parameter(torch.zeros(1, 1, config.frame_embedding_size))
        self.init_weights()
        
        bert_config = BertConfig.from_json_file('bert_config.json')
        bert_config.num_hidden_layers = 18
        self.text_encoder = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext', config=bert_config, add_pooling_layer=False)      
        self.cls_head = nn.Sequential(
                  #nn.Linear(self.text_encoder.config.hidden_size*2, self.text_encoder.config.hidden_size*2),
                  #nn.ReLU(),
                  nn.BatchNorm1d(self.text_encoder.config.hidden_size*2),
                  PositionwiseFeedForward(768*2,1024),
                  nn.Linear(self.text_encoder.config.hidden_size*2, 200)
                )            

        self.share_cross_attention(self.text_encoder.encoder)
        self.loss_fn = nn.CrossEntropyLoss()
        '''if self.distill:
            self.visual_encoder_m = AutoModel.from_config(visual_config)
            self.visual_encoder_m.embeddings.word_embeddings = nn.Identity()
            self.visual_encoder_m.pooler = nn.Identity()
            self.visual_encoder_m._init_weights(self.visual_encoder)
            self.cls_token_m = nn.Parameter(torch.zeros(1, 1, config.frame_embedding_size))
            
            ## to do fix
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False) 
            self.share_cross_attention(self.text_encoder_m.encoder)                

            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, 2)
                    )                

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995'''
            
            
    def forward(self, inputs, alpha=0, inference=False):
        B = inputs['frame_input'].shape[0]
        cls_mask = torch.ones([B,1],device='cuda')
        visual_cls = self.visual_cls.expand(B,-1,-1)
        visual_embeds = torch.cat([visual_cls,inputs['frame_input']],dim=1)
        visual_mask = torch.cat([cls_mask,inputs['frame_mask']],dim=1)
        visual_embeds = self.visual_encoder(inputs_embeds = visual_embeds,attention_mask=visual_mask)['last_hidden_state']
        visual_embeds = visual_embeds*visual_mask.unsqueeze(-1)
        #visual_embeds = self.visual_encoder(visual_embeds)
                  
        bert_embedding = self.text_encoder(inputs['title_input'], 
                                   attention_mask = inputs['title_mask'], 
                                   encoder_hidden_states = [visual_embeds],
                                   encoder_attention_mask = [visual_mask],        
                                   return_dict = True,output_hidden_states=True,
                                  )  
        mean_last_hidden = bert_embedding['last_hidden_state']
        mean_last_hidden = (mean_last_hidden*inputs['title_mask'].unsqueeze(-1)).sum(1)/inputs['title_mask'].sum(1).unsqueeze(-1)
        
        #clf_token = bert_embedding['last_hidden_state'][:,0]

        mean_last4 = bert_embedding['hidden_states'][-4:]
        mean_last4 = sum( [(hidden*inputs['title_mask'].unsqueeze(-1)).sum(1)/inputs['title_mask'].sum(1).unsqueeze(-1) for hidden in mean_last4]   ) / 4

        final_embedding =torch.cat([mean_last_hidden,mean_last4],dim=1)
        
        prediction = self.cls_head(final_embedding)
        
        #hidden_state = output.last_hidden_state[:,0,:]            
        #prediction = self.cls_head(hidden_state)
        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'],self.loss_fn)
        '''if train:
            if self.distill:  
                # to do fix variable name if using distill
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)  
                    image0_embeds_m, image1_embeds_m = torch.split(image_embeds_m,targets.size(0))
                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = [image0_embeds_m,image1_embeds_m],
                                               encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                                         image_atts[image0_embeds.size(0):]],        
                                               return_dict = True,
                                              )    
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

                loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()                        
            else:        
                loss = F.cross_entropy(prediction, inputs['label'])     
            return loss,prediction
        else:
            return prediction'''
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @staticmethod
    def cal_loss(prediction, label,loss_fn):#,level1_label,pred_level1,bce_fn):
        label = label.squeeze(dim=1)
        loss = loss_fn(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label,prediction
    def share_cross_attention(self, model):
            
        for i in range(6):
            layer_num = 6+i*2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias 