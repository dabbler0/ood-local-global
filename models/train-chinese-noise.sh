# Chinese dropout
for n in 0 0.25 0.5 0.75
do
  CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/chinese-corpus --vocab /raid/lingo/abau/lm/chinese-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/chinese-dropout-$n-2 --layers 2 --lr 3e-4 --epochs 15 --state_dropout $n --seed 2 &
done
