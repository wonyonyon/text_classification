
# Text Classification
cnn、rnn、rcnn、rcnn_attention、transformer based  pre-trained word embedding <br>
bert，albert，ernie based char and pre-trained language model

# Description
> Download pre-trained language model and `chinese news data`[THUCNews](http://thuctc.thunlp.org/)


# How to use
```
# classification based  pre-trained word embedding
python run.py --model cnn --data_conf conf/data.yaml

# bert or else based char
python run.py --model bert --data_conf conf/data.yaml

```

## Preferences
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
[2] Convolutional Neural Networks for Sentence Classification <br>
[3] Bert-Chinese-Text-Classification-Pytorch(https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch) <br>
[4] attention-is-all-you-need-pytorch(https://github.com/jadore801120/attention-is-all-you-need-pytorch) <br>
[5] Chinese-Text-Classification-Pytorch(https://github.com/649453932/Chinese-Text-Classification-Pytorch)