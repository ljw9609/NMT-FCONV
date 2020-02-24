DATADIR=data-bin/wmt18_en_zh
TRAIN=trainings/wmt18_en_zh

# Fully convolutional sequence-to-sequence model
mkdir -p $TRAIN/fconv
# CUDA_VISIBLE_DEVICES=0 fairseq-train $DATADIR --memory-efficient-fp16\
#   --arch fconv --mu 2000000 --me 500 \
#   -s en -t zh \
#   --max-tokens 2048 \
#   --lr 0.00005 --clip-norm 0.1 --dropout 0.2 --min-lr 1e-09\
#   --update-freq 1 \
#   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#   --lr-scheduler fixed --force-anneal 50 \
#   --no-epoch-checkpoints --save-interval-updates 100000 --keep-interval-updates 1\
#   --save-dir $TRAIN/fconv \

#CUDA_VISIBLE_DEVICES=0 fairseq-train $DATADIR \
#  --clip-norm 0 --optimizer adam --lr 0.0005 \
#  --source-lang en --target-lang zh --max-tokens 2048 \
#  --max-tokens 2048 \
#  --min-lr '1e-09' --weight-decay 0.0001 \
#  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#  --lr-scheduler inverse_sqrt \
#  --ddp-backend=no_c10d \
#  --max-update 500000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
#  --no-epoch-checkpoints --save-interval-updates 100000 --keep-interval-updates 1\
#  --adam-betas '(0.9, 0.98)' --keep-last-epochs 2 \
#  --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
#  -a lightconv_wmt_zh_en_big --save-dir $TRAIN/fconv \
#  --encoder-conv-type lightweight --decoder-conv-type lightweight \
#  --encoder-glu 0 --decoder-glu 0

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATADIR --memory-efficient-fp16 \
  --clip-norm 0 --optimizer adam --lr 0.0005 \
  --source-lang zh --target-lang en --max-tokens 2048 \
  --min-lr '1e-09' --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler inverse_sqrt \
  --ddp-backend=no_c10d \
  --max-epoch 5 \
  --max-update 10000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --no-epoch-checkpoints --save-interval-updates 100000 --keep-interval-updates 1\
  --adam-betas '(0.9, 0.98)' --keep-last-epochs 2 \
  --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
  -a lightconv_wmt_zh_en_big --save-dir $TRAIN/fconv \
  --encoder-conv-type lightweight --decoder-conv-type lightweight \
  --encoder-glu 0 --decoder-glu 0
