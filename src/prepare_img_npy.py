import json
import numpy as np
from tqdm import tqdm
data_dir = '../data/annotations/'
image_dir = '../data/zip_feats/'

with open(data_dir + 'labeled.json') as f:
    labeled = json.load(f)
with open(data_dir + 'test_b.json') as f:
    test = json.load(f)

train_feats = []
for i in labeled:
    this_id = i['id']
    img = np.load(image_dir+'labeled/'+i+'.npy')
    train_feats.append(img)
np.save('../data/labeled.npy',train_feats,allow_pickle=True)

test_feats = []
for i in test:
    this_id = i['id']
    img = np.load(image_dir+'test_b/'+i+'.npy')
    test_feats.append(img)
np.save('../data/testb.npy',test_feats,allow_pickle=True)

