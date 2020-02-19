DATADIR=data-bin/wmt18_en_zh
BPEDIR=data/wmt18_en_zh
MODEL=trainings/wmt18_en_zh/fconv/checkpoint_best.pt

fairseq-interactive $DATADIR \
  --memory-efficient-fp16 \
  --path $MODEL \
  -s en -t zh \
  --tokenizer moses \
  --bpe subword_nmt \
  --bpe-codes $BPEDIR/codes.32000.bpe.en \
  --remove-bpe \
  --print-alignment \
  --beam 10
