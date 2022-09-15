# Finnish
# Embed
#for n in 0 0.25 0.5 0.75
#do
#  CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-embed-$n-1 --layers 2 --lr 3e-4 --epochs 15 --embed_dropout $n --seed 1 &
#done

#for n in 0 0.25 0.5 0.75
#do
#  CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-embed-$n-2 --layers 2 --lr 3e-4 --epochs 15 --embed_dropout $n --seed 2 &
#done

# Subst
#for n in 0 0.25 0.5 0.75
#do
#  CUDA_VISIBLE_DEVICES=12 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-swap-$n-1 --layers 2 --lr 3e-4 --epochs 15 --subst_prob $n --seed 1 &
#done

#for n in 0 0.25 0.5 0.75
#do
#  CUDA_VISIBLE_DEVICES=13 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-swap-$n-2 --layers 2 --lr 3e-4 --epochs 15 --subst_prob $n --seed 2 &
#done

# Small English subst
for n in 0 0.25 0.5 0.75
do
  CUDA_VISIBLE_DEVICES=12 python train.py --data /raid/lingo/abau/lm/20k-corpus --vocab /raid/lingo/abau/lm/20k-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/20k-subst-$n-1 --layers 2 --lr 3e-4 --epochs 15 --subst_prob $n --seed 1 &
done

for n in 0 0.25 0.5 0.75
do
  CUDA_VISIBLE_DEVICES=13 python train.py --data /raid/lingo/abau/lm/20k-corpus --vocab /raid/lingo/abau/lm/20k-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/20k-subst-$n-2 --layers 2 --lr 3e-4 --epochs 15 --subst_prob $n --seed 2 &
done

# Finnish subst
for n in 0 0.25 0.5 0.75
do
  CUDA_VISIBLE_DEVICES=14 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-subst-$n-1 --layers 2 --lr 3e-4 --epochs 15 --subst_prob $n --seed 1 &
done

for n in 0 0.25 0.5 0.75
do
  CUDA_VISIBLE_DEVICES=15 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-subst-$n-2 --layers 2 --lr 3e-4 --epochs 15 --subst_prob $n --seed 2 &
done

# Finnish
#for n in 0 0.25 0.5 0.75
#do
#  CUDA_VISIBLE_DEVICES=12 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-state-$n-1 --layers 2 --lr 3e-4 --epochs 15 --state_dropout $n --seed 1 &
#done

#for n in 0 0.25 0.5 0.75
#do
#  CUDA_VISIBLE_DEVICES=13 python train.py --data /raid/lingo/abau/lm/finnish-corpus --vocab /raid/lingo/abau/lm/finnish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/finnish-state-$n-2 --layers 2 --lr 3e-4 --epochs 15 --state_dropout $n --seed 2 &
#done
