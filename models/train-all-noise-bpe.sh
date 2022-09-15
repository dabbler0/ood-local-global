# Finnish
# =======
for s in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/finnish-bpe-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-seed-$s --layers 2 --lr 3e-4 --epochs 15 --seed $s &
done

# English
# =======
for s in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=7 python train.py --data /raid/lingo/abau/lm/20k-bpe-corpus --vocab /raid/lingo/abau/lm/20k-bpe-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/20k-seed-$s --layers 2 --lr 3e-4 --epochs 15 --seed $s &
done
