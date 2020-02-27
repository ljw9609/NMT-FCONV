DATADIR=data-bin/wmt18_en_zh
BPEDIR=data/wmt18_en_zh
#MODEL=trainings/wmt18_en_zh/fconv/checkpoint_best.pt
MODEL=./checkpoint_best.pt

fairseq-interactive $DATADIR \
  --path $MODEL \
  -s en -t zh \
  --tokenizer moses \
  --bpe subword_nmt \
  --bpe-codes $BPEDIR/codes.32000.bpe.en \
  --remove-bpe \
  --print-alignment \
  --beam 5

#fairseq-interactive \
#    $DATADIR \
#    --memory-efficient-fp16 \
#    --path $MODEL \
#    -s zh -t en \
#    --tokenizer moses \
#    --bpe subword_nmt \
#    --bpe-codes $BPEDIR/codes.32000.bpe.zh \
#    --beam 5 --remove-bpe
