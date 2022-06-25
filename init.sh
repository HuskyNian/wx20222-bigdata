pip install -r requirements.txt
mkdir data/zip_feats/test_b
unzip -n data/zip_feats/test_b.zip -d data/zip_feats/test_b/
mkdir data/zip_feats/labeled
unzip -n data/zip_feats/labeled.zip -d data/zip_feats/labeled/
mkdir data/zip_feats/unlabeled
unzip -n data/zip_feats/unlabeled.zip -d data/zip_feats/unlabeled/
mkdir data/zip_feats/clustered_feats

mkdir pretrained_models
mkdir finetune_models
cd src
python prepare_img_npy.py

python split_pretrain_data.py


