TEXT=data/wmt18_en_zh
DATADIR=data-bin/wmt18_en_zh
TRAIN=trainings/wmt18_en_zh

# Fully convolutional sequence-to-sequence model
mkdir -p $TRAIN/fconv
CUDA_VISIBLE_DEVICES=0 fairseq-train $DATADIR --memory-efficient-fp16\
  --arch fconv --mu 1000000 --me 50 \
  --batch-size 16 --max-tokens 2048 \
  --lr 0.5 --clip-norm 0.1 --dropout 0.2 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --no-epoch-checkpoints --save-interval-updates 100000 --keep-interval-updates 1\
  --restore-file \
  --save-dir TRAIN/fconv \
