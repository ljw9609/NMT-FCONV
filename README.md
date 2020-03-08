# NMT_FCONV

FCONV Model part of the course project of Northwestern University COMP_SCI 496: Statistical Machine Learning.


### Input & Output

+ Input: Sentences or paragraphs in source language. (eg: Slowly and not without struggle, America began to listen.)

+ Output: Sentences or paragraphs in target language. (eg: 美国缓慢地开始倾听，但并非没有艰难曲折。)

## Core Model

This is the Fully Convolutional Seq-to-Seq Model of the Machine Translation project. We train the model from scratch.

## Deliverables

+ Data Preprocessing Script [view](https://github.com/ljw9609/NMT-FCONV/blob/master/FCONV/data_preprocess.sh)
+ Model Training Script [view](https://github.com/ljw9609/NMT-FCONV/blob/master/FCONV/train.sh)
+ Trained model
  + EN->ZH [Download](https://drive.google.com/open?id=1QOMDSDuGZy_gs6KWbm0TFXV2wQbKk1C6)
  + ZH->EN [Download](https://drive.google.com/open?id=1QbLi7bRewaLRAPj-1M9380yGEC36Ssuf)
+ Model Inference [View](https://github.com/ljw9609/NMT-FCONV/blob/master/src/nmt.py)
+ Web API [View](https://github.com/ljw9609/NMT-FCONV/blob/master/src/app.py)
+ Dockerfile [View](https://github.com/ljw9609/NMT-FCONV/blob/master/Dockerfile)
+ Docker Image [View](https://hub.docker.com/repository/docker/ljw96/nmt-fconv)

## Install

```py
# install python libraries
pip install -r requirements.txt

# install subword-nmt & mosesdecoder
git clone https://github.com/rsennrich/subword-nmt

git clone https://github.com/moses-smt/mosesdecoder
```

## Data Preprocessing

### Data collection

We use *wmt18-news commentary v13* data set.

### Data cleaning

+ Word tokenization: `nltk` for English and `jieba` for Chinese
+ Casing: Convert all sentences to lower case
+ Merge blank lines: Merge useless blank lines in the data set

### Data preprocessing

Preprocessing includes 3 parts:
+ Collect, unzip, tokenize, clean data set in `FCONV/preprocess/preprocess.py`
+ Build vocabularies and bpe codes using [subword-nmt.apply_bpe](https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py)
+ Binarize data set using `fairseq-preprocess`

Preprocessing shell script:

```sh
chmod 777 ./FCONV/data_preprocess.sh

./FCONV/data_preprocess.sh
```

## Model Training

We train the model with the Fully Convolutional Seq-to-Seq model in `fairseq.model.fconv`, using the prepared data set.

On an AWS 'g4dn.xlarge' instance, the training should last for about 10 hour, 55 epochs per direction.

**Attention**: The training script only includes ZH to EN model, you need to modify the `source-lang` and `target-lang` for EN to ZH.

Training script:

```py
chmod 777 ./FCONV/train.sh

./FCONV/train.sh
```

## Usage

### Download Model
Move or download the model file into the `FCONV/model` directory, the dictionary and bpe codes in that directory are learnt from the data set we prepared. If you want to use your own data set, please replace these files accordingly.

### Model Inference

```py
PATH = './FCONV/model'

en2zh = FConvModel.from_pretrained(
    PATH,
    checkpoint_file='model_en2zh.pt',
    data_name_or_path=PATH,
    tokenizer='moses',
    bpe='subword_nmt',
    bpe_codes=PATH + '/en.code'
)
en2zh.translate('Slowly and not without struggle, America began to listen.')
```

### Start server

#### 1. Run on local

```sh
# run the server
gunicorn --config ./conf/gunicorn_config.py src:app
```

#### 2. Run in Docker

```sh
# build image
docker build -t nmt-mass .

# start a container
docker run -p 4869:8000 --name mass nmt-mass
```

### Web API

#### HTTP Request

```
POST /translate
Host: YOUR_SERVER_ADDRESS
Body: {
  's_lang': 'en',
  't_lang': 'zh',
  's_text': 'Slowly and not without struggle, America began to listen.'
}
```

#### HTTP Response

```
{
  's_text': 'Slowly and not without struggle, America began to listen.',
  't_text': '美国缓慢地开始倾听，但并非没有艰难曲折。'
}
```

## Reference
[Fully Convolutional Seq-to-Seq](https://github.com/pytorch/fairseq/tree/master/examples/conv_seq2seq)
