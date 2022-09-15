CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/turkish-corpus --vocab /raid/lingo/abau/lm/turkish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/turkish-1 --layers 2 --lr 3e-4 --epochs 15 --seed 1 &
CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/turkish-corpus --vocab /raid/lingo/abau/lm/turkish-vocab.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/turkish-2 --layers 2 --lr 3e-4 --epochs 15 --seed 2 &

#for n in 0.2 0.4 0.6 0.8
#do
#  CUDA_VISIBLE_DEVICES=6 python train.py --data /raid/lingo/abau/lm/training-monolingual/news.2007.en.1m.tokenized --vocab /raid/lingo/abau/lm/vocab.en.1m.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/en.1m.2layer.lr1e6.fixed.embedding_dropout$n --layers 2 --lr 1e-4 --embed_dropout $n --epochs 100 &
#done

#for n in 0.1 0.3 0.5 0.7
#do
#  CUDA_VISIBLE_DEVICES=7 python train.py --data /raid/lingo/abau/lm/training-monolingual/news.2007.en.1m.tokenized --vocab /raid/lingo/abau/lm/vocab.en.1m.pt --vocab_size 16384 --output /raid/lingo/abau/lm/monolingual-models/en.1m.2layer.lr1e6.fixed.embedding_dropout$n --layers 2 --lr 1e-4 --embed_dropout $n --epochs 100 &
#done
