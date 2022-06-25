# wbdc2022-preliminary-b725af20e53b4273b21875231f3c2a04
wx2022_bigdata

# 2022微信大数据初赛 竹竹天下第一队伍代码，队伍id:b725af20e53b4273b21875231f3c2a04

# 环境配置
python:3.7.13  pytorch:1.11.0+cu113
requirements.txt

# 数据
使用了官方提供的无标签及有标签数据进行预训练，用有标签数据进行微调，共100 + 10 + 2.5 = 112.5w 数据

# 预训练模型
使用了huggingface 提供的 'hfl/chinese-roberta-wwm-ext' 模型， 链接：https://huggingface.co/hfl/chinese-roberta-wwm-ext

# 算法描述
1. 使用112.5w数据进行mlm,mfm,itm预训练10ep，并使用10w有标签数据进行微调
2. 更改了模型里面多个细节，如embed层交互，后四层pooling  concat  最后一层pooling 加快收敛
3. 使用两个bertembedding 对视频和文本特征做不同的embed使预训练能发挥最大作用
4. 在title,ocr,asr文本优先选取最有作用的title，其实是ocr最后是asr因为如果只有背景音乐的时候asr是一堆乱码，使用动态截取，title,ocr,asr都保证有最少的长度，但是比如ocr长度为0，那么title
或者asr就可以多截取一点，最后总长度为256，充分利用好数据
5. 单模全量线上0.680， 21折以后线上0.690
6. 使用对抗训练和ema

# 训练流程
先运行pretrained_model.py预训练，其次直接运行main.py进行微调

# 测试流程
10折 + 10折 + 全量
